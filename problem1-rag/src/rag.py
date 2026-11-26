"""RAG core: retrieve top-k chunks, then generate a grounded answer.

Grounding rules enforced in the prompt and in code:
- the model may only use the supplied chunks and must cite chunk ids;
- if retrieval returns nothing above the score floor, we answer
  "I don't have relevant context" WITHOUT calling the LLM - a grounded
  refusal is cheaper and cannot hallucinate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLMClient
from .embeddings import Embedder
from .store import VectorStore

log = logging.getLogger("rag")

# Cosine-similarity floor below which chunks are treated as irrelevant.
# Calibrated on the eval set; see eval/results.md.
MIN_SCORE = 0.25

NO_CONTEXT_ANSWER = (
    "I don't have relevant context in the corpus to answer that question."
)

PROMPT_TEMPLATE = """You are a question-answering service over a document corpus.
Answer the question using ONLY the context chunks below. Cite the chunk id
of every chunk you rely on, in square brackets, e.g. [a1b2c3]. If the chunks
do not contain the answer, say exactly: "{no_context}"

Context chunks:
{context}

Question: {question}

Answer (with chunk citations):"""


@dataclass
class RAGResponse:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved: int = 0
    retrieved_keys: list[str] = field(default_factory=list)  # "source:position" per hit
    retrieval_ms: float = 0.0
    had_context: bool = True
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model, settings.embedding_dim)
        self.store = VectorStore(settings.index_dir, settings.embedding_dim)
        self.llm = LLMClient(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)

    def query(
        self, question: str, k: int | None = None, metadata_filter: dict | None = None
    ) -> RAGResponse:
        t0 = time.perf_counter()
        k = k or self.settings.top_k
        qvec = self.embedder.embed_one(question)
        hits = self.store.search(qvec, k, metadata_filter)
        hits = [h for h in hits if h["score"] >= MIN_SCORE]
        keys = [f"{h['source']}:{h['position']}" for h in hits]
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not hits:
            resp = RAGResponse(
                question=question,
                answer=NO_CONTEXT_ANSWER,
                had_context=False,
                retrieval_ms=retrieval_ms,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            self._log(resp)
            return resp

        context = "\n\n".join(
            f"[{h['chunk_id'][:6]}] (source: {h['source']})\n{h['text']}" for h in hits
        )
        msg = self.llm.chat(
            model=self.settings.llm_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(
                        no_context=NO_CONTEXT_ANSWER, context=context, question=question
                    ),
                }
            ],
        )
        answer = msg.text
        cited = [h["chunk_id"][:6] for h in hits if h["chunk_id"][:6] in answer]
        resp = RAGResponse(
            question=question,
            answer=answer,
            citations=cited,
            retrieved=len(hits),
            retrieved_keys=keys,
            retrieval_ms=retrieval_ms,
            had_context=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=msg.prompt_tokens,
            completion_tokens=msg.completion_tokens,
        )
        self._log(resp)
        return resp

    @staticmethod
    def _log(resp: RAGResponse) -> None:
        # per-query structured log: latency, chunk count, token usage
        log.info(
            "query",
            extra={
                "latency_ms": round(resp.latency_ms, 1),
                "chunks": resp.retrieved,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "had_context": resp.had_context,
            },
        )
