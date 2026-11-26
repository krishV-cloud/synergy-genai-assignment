"""CLI: chunk the corpus, embed it, and upsert into the vector store.

Usage: python scripts/ingest.py
Idempotent: chunk ids are content-addressed, so re-runs skip existing chunks.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.embeddings import Embedder
from src.ingestion import ingest_corpus
from src.store import VectorStore

def main() -> None:
    settings = Settings.from_env()
    chunks = ingest_corpus(settings.corpus_dir, settings.chunk_size, settings.chunk_overlap)
    print(f"chunked corpus: {len(chunks)} chunks from {settings.corpus_dir}")
    embedder = Embedder(settings.embedding_model, settings.embedding_dim)
    store = VectorStore(settings.index_dir, settings.embedding_dim)
    vectors = embedder.embed([c.text for c in chunks])
    added = store.add_chunks(chunks, vectors)
    print(f"store upsert: {added} new, {store.size} total (index: {settings.index_dir})")

if __name__ == "__main__":
    main()
