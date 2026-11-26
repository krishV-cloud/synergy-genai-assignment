"""HTTP API for the RAG service.

Run:  uvicorn src.api:app --reload
Then: POST /query  {"question": "...", "k": 5, "metadata_filter": {"doc_type": "pdf"}}
      POST /ingest (re-ingests the corpus idempotently)
      GET  /health
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .config import Settings
from .ingestion import ingest_corpus
from .rag import RAGService

settings = Settings.from_env()
service = RAGService(settings)
app = FastAPI(title="Cost-efficient RAG service")


class QueryRequest(BaseModel):
    question: str
    k: int | None = None
    metadata_filter: dict | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": service.store.size}


@app.post("/ingest")
def ingest() -> dict:
    chunks = ingest_corpus(settings.corpus_dir, settings.chunk_size, settings.chunk_overlap)
    vectors = service.embedder.embed([c.text for c in chunks]) if chunks else []
    added = service.store.add_chunks(chunks, vectors) if chunks else 0
    return {"corpus_chunks": len(chunks), "newly_added": added, "total": service.store.size}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    resp = service.query(req.question, req.k, req.metadata_filter)
    return {
        "answer": resp.answer,
        "citations": resp.citations,
        "retrieved_chunks": resp.retrieved,
        "had_context": resp.had_context,
        "latency_ms": round(resp.latency_ms, 1),
        "tokens": {"prompt": resp.prompt_tokens, "completion": resp.completion_tokens},
    }
