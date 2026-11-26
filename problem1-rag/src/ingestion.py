"""Corpus ingestion: PDF/HTML/MD -> chunks.

Idempotent: every chunk carries a stable content hash (doc path + position +
text), and re-ingesting the same corpus skips hashes already in the store,
so no duplicate vectors are ever written.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import markdown as md_lib
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: str  # sha1 of (source, position, text) - stable across runs
    source: str  # relative path of the origin document
    doc_type: str  # pdf | html | md  (usable as a metadata filter)
    position: int  # chunk index within the document
    text: str
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(source: str, position: int, text: str) -> str:
        return hashlib.sha1(f"{source}::{position}::{text}".encode()).hexdigest()


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _load_md(path: Path) -> str:
    html = md_lib.markdown(path.read_text(encoding="utf-8", errors="ignore"))
    return BeautifulSoup(html, "html.parser").get_text(separator="\n")


LOADERS = {".pdf": _load_pdf, ".html": _load_html, ".htm": _load_html, ".md": _load_md}


def _normalise(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-window chunking, breaking on paragraph/sentence boundaries.

    Defaults come from .env (800 chars / 120 overlap): small enough that a
    chunk usually holds one idea, large enough to keep local context.
    """
    text = _normalise(text)
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start + chunk_size // 2, end)
            if boundary == -1:
                boundary = text.rfind(". ", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
        if start >= len(text):
            break
    return [c for c in chunks if c]


def ingest_corpus(corpus_dir: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Load every supported file under corpus_dir and return all chunks."""
    root = Path(corpus_dir)
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None or not path.is_file():
            continue
        rel = str(path.relative_to(root))
        for i, piece in enumerate(chunk_text(loader(path), chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(rel, i, piece),
                    source=rel,
                    doc_type=path.suffix.lower().lstrip("."),
                    position=i,
                    text=piece,
                    metadata={"source": rel, "doc_type": path.suffix.lower().lstrip(".")},
                )
            )
    return chunks
