# Problem 1 - Cost-Efficient RAG Application

A QA service over a document corpus, backed by FAISS (embedded, zero-infra
vector store), with honest evaluation of retrieval quality, answer quality,
latency and cost.

## Architecture

```
PDF/HTML/MD corpus
      |  ingestion.py  (chunk: 800 chars, 120 overlap; sha1-idempotent)
      v
embeddings.py  (all-MiniLM-L6-v2, 384-dim, normalised)
      v
store.py  (FAISS IndexFlatIP + JSONL metadata sidecar)
      v
rag.py  (top-k -> grounded answer with [chunk-id] citations;
         no relevant context -> grounded refusal, no LLM call)
      v
api.py  (FastAPI: POST /query, POST /ingest, GET /health)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
# put corpus files in ./data/corpus (pdf/html/md)
uvicorn src.api:app --reload
curl -X POST localhost:8000/ingest
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"question": "...", "k": 5, "metadata_filter": {"doc_type": "pdf"}}'
```

## Why FAISS (store choice)

The brief's cost problem is that managed vector DBs bill for always-on
capacity that scales with stored vectors. FAISS removes the server entirely:
it is an embedded library, so marginal infra cost is the RAM of the API
process. Exact inner-product search on normalised vectors gives cosine
similarity with no approximation loss. Trade-offs and the point at which a
managed DB wins again are quantified in costs.md.

## Evaluation

`eval/harness.py` runs the fixed question set (`eval/questions.json`) and
writes `eval/results.json`: Recall@k, Hit Rate, MRR, nDCG@k, context
precision, faithfulness/relevance (LLM-as-judge, reusing Problem 2's
pipeline), p50/p95 latency, and per-question token usage.

## Discussion

(when would you switch back to managed; was retrieval or generation the
weak link - filled in with measured results)
