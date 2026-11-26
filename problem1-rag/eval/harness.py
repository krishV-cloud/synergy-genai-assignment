"""Evaluation harness for the RAG service.

Runs a fixed question set (eval/questions.json, 15-30 items) and reports:
- retrieval: Recall@k, Hit Rate, MRR, nDCG@k, context precision
- answer:    faithfulness + relevance (LLM-as-judge), EM/F1 where gold answers exist
- latency:   p50 / p95 of retrieval-only and end-to-end time
- cost:      token usage per question (the infra cost model lives in costs.md)

Ground truth format per question:
{
  "question": "...",
  "evidence": ["verbatim substring from the relevant chunk(s)", ...],
  "gold_answer": "..."            # optional, enables EM/F1
}

Evidence substrings are matched against the stored chunks at eval time, so
labels stay correct even if chunking parameters change. A question whose
evidence matches nothing fails loudly instead of silently scoring zero.
The judge model is a DIFFERENT family than the answer model (JUDGE_MODEL vs
LLM_MODEL) to keep self-enhancement bias out of the answer metrics.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import json
import math
import re
import statistics
from pathlib import Path
import time
from pathlib import Path

from src.config import Settings
from src.llm import LLMClient
from src.rag import RAGService, NO_CONTEXT_ANSWER as NO_CONTEXT

K = 5  # matches default TOP_K; reported metrics are @K


# ---------- retrieval metrics ----------

def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    return len(set(retrieved) & set(relevant)) / len(relevant) if relevant else 0.0


def hit_rate(retrieved: list[str], relevant: list[str]) -> float:
    return 1.0 if set(retrieved) & set(relevant) else 0.0


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    dcg = sum(
        (1.0 if doc in relevant else 0.0) / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def context_precision(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of retrieved chunks that are relevant (precision@k)."""
    return len(set(retrieved) & set(relevant)) / len(retrieved) if retrieved else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, math.ceil(p / 100 * len(values)) - 1)
    return values[idx]


# ---------- answer metrics ----------

def token_f1(prediction: str, gold: str) -> float:
    p, g = prediction.lower().split(), gold.lower().split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(prediction: str, gold: str) -> float:
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    return 1.0 if norm(prediction) == norm(gold) else 0.0


JUDGE_PROMPT = """You are grading an answer from a retrieval-augmented QA system.

Question: {question}
Retrieved context (what the system saw):
{context}

System's answer: {answer}
{gold_line}
Score two dimensions, each 0-2:
- faithfulness: 2 = every claim is supported by the context, 1 = minor unsupported detail, 0 = contradicts or invents content. If the system said it has no relevant context, score 2 when the context truly lacks the answer, else 0.
- relevance: 2 = answers the question directly, 1 = partial, 0 = off-topic.

Reply with ONLY a JSON object: {{"faithfulness": <0|1|2>, "relevance": <0|1|2>, "rationale": "<one sentence>"}}"""


def judge_answer(llm: LLMClient, model: str, question: str, context: str, answer: str, gold: str | None) -> dict:
    gold_line = f"Reference answer: {gold}" if gold else ""
    resp = llm.chat(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, context=context, answer=answer, gold_line=gold_line)}],
    )
    m = re.search(r"\{.*\}", resp.text, re.DOTALL)
    try:
        verdict = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        verdict = {}
    return {
        "faithfulness": verdict.get("faithfulness"),
        "relevance": verdict.get("relevance"),
        "rationale": verdict.get("rationale", ""),
        "judge_parse_ok": bool(verdict),
    }


def run_eval(settings: Settings, judge_model: str | None = None) -> dict:
    service = RAGService(settings)
    judge_model = judge_model or settings.judge_model
    llm = LLMClient(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)
    questions = json.loads(Path("eval/questions.json").read_text())

    # map evidence substrings -> stored chunk keys (ground truth resolution)
    chunk_lookup = service.store.meta
    def relevant_keys(item: dict) -> list[str]:
        keys = set()
        for ev in item.get("evidence", []):
            ev_norm = re.sub(r"\s+", " ", ev.lower())
            for rec in chunk_lookup:
                if ev_norm in re.sub(r"\s+", " ", rec["text"].lower()):
                    keys.add(f"{rec['source']}:{rec['position']}")
        return sorted(keys)

    per_question, lat_e2e, lat_ret = [], [], []
    done_questions = set()
    partial_path = Path("eval/results.partial.json")
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text())
            per_question = prev.get("per_question", [])
            done_questions = {q["question"] for q in per_question}
            for q in per_question:
                lat_e2e.append(q["latency_ms"])
        except Exception:
            per_question = []
    for i, item in enumerate(questions, 1):
        if item["question"] in done_questions:
            print(f"[{i}/{len(questions)}] skip (done): {item['question'][:50]}")
            continue
        relevant = relevant_keys(item)
        if not relevant and item.get("evidence"):
            raise RuntimeError(f"Q{i} evidence matched no chunks: {item['question'][:60]}")

        try:
            resp = service.query(item["question"])
        except Exception as e:
            per_question.append({"question": item["question"], "error": str(e)[:200]})
            print(f"[{i}/{len(questions)}] FAILED: {item['question'][:50]}")
            continue
        retrieved = resp.retrieved_keys[:K]
        lat_e2e.append(resp.latency_ms)
        lat_ret.append(resp.retrieval_ms)

        context_text = "\n".join(
            rec["text"] for rec in service.store.meta
            if f"{rec['source']}:{rec['position']}" in retrieved
        )
        gold = item.get("gold_answer")
        verdict = judge_answer(llm, judge_model, item["question"], context_text, resp.answer, gold)

        # retrieval metrics only where ground truth exists; abstention probes
        # are scored on whether the service refused instead of hallucinating
        retrieval_metrics = None
        if relevant:
            retrieval_metrics = {
                "recall@k": round(recall_at_k(retrieved, relevant), 3),
                "hit": hit_rate(retrieved, relevant),
                "mrr": round(mrr(retrieved, relevant), 3),
                "ndcg@k": round(ndcg_at_k(retrieved, relevant, K), 3),
                "context_precision": round(context_precision(retrieved, relevant), 3),
            }
        abstain_ok = None
        if item.get("expect_abstain"):
            abstain_ok = (not resp.had_context) or (NO_CONTEXT in resp.answer)

        per_question.append({
            "question": item["question"],
            "answer": resp.answer,
            "had_context": resp.had_context,
            "expect_abstain": bool(item.get("expect_abstain")),
            "abstain_correct": abstain_ok,
            "retrieval": retrieval_metrics,
            "answer_eval": {
                **verdict,
                **({"em": exact_match(resp.answer, gold), "f1": round(token_f1(resp.answer, gold), 3)} if gold else {}),
            },
            "latency_ms": round(resp.latency_ms, 1),
            "tokens": {"prompt": resp.prompt_tokens, "completion": resp.completion_tokens},
        })
        print(f"[{i}/{len(questions)}] {item['question'][:60]}")
        Path("eval/results.partial.json").write_text(json.dumps(
            {"completed": len(per_question), "per_question": per_question}, indent=2))

    scored = [q for q in per_question if q.get("retrieval")]
    agg = lambda f: round(statistics.mean(q["retrieval"][f] for q in scored), 3) if scored else None
    judged = [q["answer_eval"] for q in per_question if q.get("answer_eval") and q["answer_eval"]["judge_parse_ok"]]
    report = {
        "n_questions": len(questions),
        "k": K,
        "retrieval": {
            "n_scored": len(scored),
            "recall@k": agg("recall@k"),
            "hit_rate": agg("hit"),
            "mrr": agg("mrr"),
            "ndcg@k": agg("ndcg@k"),
            "context_precision": agg("context_precision"),
        },
        "answer": {
            "judge_model": judge_model,
            "faithfulness_mean": round(statistics.mean(j["faithfulness"] for j in judged), 3) if judged else None,
            "relevance_mean": round(statistics.mean(j["relevance"] for j in judged), 3) if judged else None,
            "judge_parse_failures": len(per_question) - len(judged),
            "em_mean": round(statistics.mean(q["answer_eval"]["em"] for q in per_question if q.get("answer_eval") and "em" in q["answer_eval"]), 3) if any(q.get("answer_eval") and "em" in q["answer_eval"] for q in per_question) else None,
            "f1_mean": round(statistics.mean(q["answer_eval"]["f1"] for q in per_question if q.get("answer_eval") and "f1" in q["answer_eval"]), 3) if any(q.get("answer_eval") and "f1" in q["answer_eval"] for q in per_question) else None,
        },
        "latency_ms": {
            "retrieval_p50": round(percentile(lat_ret, 50), 1),
            "retrieval_p95": round(percentile(lat_ret, 95), 1),
            "e2e_p50": round(percentile(lat_e2e, 50), 1),
            "e2e_p95": round(percentile(lat_e2e, 95), 1),
        },
        "token_totals": {
            "prompt": sum(q["tokens"]["prompt"] for q in per_question),
            "completion": sum(q["tokens"]["completion"] for q in per_question),
        },
        "abstention": {
            "n_probes": sum(1 for q in per_question if q.get("expect_abstain")),
            "correct": sum(1 for q in per_question if q.get("abstain_correct")),
        },
        "per_question": per_question,
    }
    Path("eval/results.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = run_eval(Settings.from_env())
    summary = {k: v for k, v in rep.items() if k != "per_question"}
    print(json.dumps(summary, indent=2))
