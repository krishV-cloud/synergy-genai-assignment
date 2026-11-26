"""FAISS-backed vector store with metadata sidecar.

Why FAISS (choice justification, also in README):
- zero infra cost: embedded library, no server, no pods, no per-vector billing -
  the managed-DB cost problem in the brief simply does not exist here;
- exact inner-product search on normalised vectors (= cosine), fast at our
  scale (p50 well under 10 ms at 100K vectors; measured in eval);
- persisted as two files (index + JSONL metadata), so the store is portable
  and versionable.

Trade-off accepted: no built-in metadata filtering, so we over-fetch and
filter in Python. At 100K-1M vectors that costs a few ms per query; at 10M
we would move to FAISS IVF partitioning or pgvector (see costs.md).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .ingestion import Chunk


class VectorStore:
    def __init__(self, index_dir: str, dim: int):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._index_path = self.dir / "vectors.faiss"
        self._meta_path = self.dir / "metadata.jsonl"
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[dict] = []
        self.known_ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._index_path.exists():
            self.index = faiss.read_index(str(self._index_path))
        if self._meta_path.exists():
            for line in self._meta_path.read_text().splitlines():
                rec = json.loads(line)
                self.meta.append(rec)
                self.known_ids.add(rec["chunk_id"])

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Idempotent insert: chunks whose id is already stored are skipped."""
        fresh = [(c, v) for c, v in zip(chunks, vectors) if c.chunk_id not in self.known_ids]
        if not fresh:
            return 0
        vecs = np.stack([v for _, v in fresh])
        self.index.add(vecs)
        with self._meta_path.open("a") as fh:
            for chunk, _ in fresh:
                rec = asdict(chunk)
                rec["embedded_at"] = int(time.time())
                fh.write(json.dumps(rec) + "\n")
                self.meta.append(rec)
                self.known_ids.add(chunk.chunk_id)
        faiss.write_index(self.index, str(self._index_path))
        return len(fresh)

    def search(
        self, query_vec: np.ndarray, k: int, metadata_filter: dict | None = None
    ) -> list[dict]:
        """Top-k search; optional equality filter on metadata fields.

        Over-fetches 4x when a filter is active, then filters client-side.
        """
        if self.index.ntotal == 0:
            return []
        fetch_k = k * 4 if metadata_filter else k
        fetch_k = min(fetch_k, self.index.ntotal)
        scores, idxs = self.index.search(query_vec.reshape(1, -1), fetch_k)
        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            rec = dict(self.meta[idx])
            rec["score"] = float(score)
            if metadata_filter and not all(
                rec.get(key) == val for key, val in metadata_filter.items()
            ):
                continue
            hits.append(rec)
            if len(hits) == k:
                break
        return hits

    @property
    def size(self) -> int:
        return self.index.ntotal
