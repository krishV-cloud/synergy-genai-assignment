# Cost analysis: FAISS (embedded) vs managed vector DB

Assumptions stated up front:
- 384-dim vectors (all-MiniLM-L6-v2), float32 -> 1.5 KB per vector raw.
- Managed comparison baseline: a fully managed vector DB with always-on
  capacity billed per stored vector / pod-hour (typical entry tier).
- Self-hosted FAISS runs on the same VM as the API (no extra infra) or a
  ~$5/mo small VPS; embeddings are computed once at ingest.
- Numbers to be measured/verified in eval; this table is the skeleton.

| Scale    | Managed DB (est. $/mo) | FAISS self-host (est. $/mo) | Notes |
|----------|------------------------|-----------------------------|-------|
| 100K vectors   | TBD | ~0-5 (shared VM) | RAM ~0.2 GB |
| 1M vectors     | TBD | ~5-10 (small VPS) | RAM ~2 GB |
| 10M vectors    | TBD | ~40-80 (RAM-heavy VM) | RAM ~20 GB; consider IVF/pgvector |

Latency: p50/p95 retrieval measured in eval/results.json.
Trade-offs accepted: no managed replication/backups, client-side metadata
filtering, single-node. When to switch back to managed: multi-region reads,
>10M vectors with tight p99, or no ops capacity - see README Discussion.
