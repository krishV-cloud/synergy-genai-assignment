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
| 100K vectors   | see below | ~0-5 (shared VM) | RAM ~0.2 GB |
| 1M vectors     | see below | ~5-10 (small VPS) | RAM ~2 GB |
| 10M vectors    | see below | ~40-80 (RAM-heavy VM) | RAM ~20 GB; consider IVF/pgvector |

Latency: p50/p95 retrieval measured in eval/results.json.
Trade-offs accepted: no managed replication/backups, client-side metadata
filtering, single-node. When to switch back to managed: multi-region reads,
>10M vectors with tight p99, or no ops capacity - see README Discussion.


## Managed pricing (Pinecone, checked Nov 27 2026: pinecone.io/pricing)
Starter free up to 2 GB; Builder $20/mo; Standard $50/mo minimum; storage from $0.33/GB/mo; reads from $16/M read units (1 RU per GB per query).
100K and 1M vectors (0.15 / 1.5 GB at 384d float32) fit the free Starter tier, so FAISS has no cost edge there. At 10M (15 GB) managed is about $50/mo (minimum) vs an estimated $40-80 for a RAM-heavy VM: roughly break-even.
