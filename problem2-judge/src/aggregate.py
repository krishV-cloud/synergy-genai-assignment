"""Suite aggregation: per-case verdicts -> suite report + A/B winner."""
from __future__ import annotations

import statistics
from collections import defaultdict

from .schema import Verdict


def pointwise_report(verdicts: list[Verdict], pass_overall: float = 3.5) -> dict:
    ok = [v for v in verdicts if v.parse_ok and v.overall is not None]
    report = {
        "n_cases": len(verdicts),
        "parse_failures": len(verdicts) - len(ok),
        "pass_rate": round(sum(1 for v in ok if v.overall >= pass_overall) / len(ok), 3) if ok else None,
        "mean_overall": round(statistics.mean(v.overall for v in ok), 3) if ok else None,
        "per_criterion_mean": {},
    }
    criteria = defaultdict(list)
    for v in ok:
        for c, s in v.scores.items():
            criteria[c].append(s)
    report["per_criterion_mean"] = {c: round(statistics.mean(s), 3) for c, s in criteria.items()}
    return report


def pairwise_report(verdict_pairs: list[list[Verdict]]) -> dict:
    """Each item is [AB verdict, BA verdict] for one case.

    Position bias check: a flip is AB-says-A-wins but BA-says-B-wins (i.e. the
    judge prefers whichever answer it saw first/last inconsistently). Only
    order-consistent verdicts count toward the win rate.
    """
    flips = 0
    wins = {"A": 0, "B": 0, "tie": 0}
    n_ok = 0
    for pair in verdict_pairs:
        if len(pair) != 2 or not all(v.parse_ok for v in pair):
            continue
        n_ok += 1
        w1, w2 = pair[0].winner, pair[1].winner
        if w1 != w2:
            flips += 1
            wins["tie"] += 1  # disagreement collapses to tie
        else:
            wins[w1] += 1
    decided = wins["A"] + wins["B"]
    winner = None
    if decided:
        if wins["A"] > wins["B"]:
            winner = "A"
        elif wins["B"] > wins["A"]:
            winner = "B"
        else:
            winner = "tie"
    return {
        "n_cases": len(verdict_pairs),
        "n_judged": n_ok,
        "position_flip_rate": round(flips / n_ok, 3) if n_ok else None,
        "wins": wins,
        "win_rate_A": round(wins["A"] / n_ok, 3) if n_ok else None,
        "winner": winner,
    }


def compare_configs(verdicts_by_config: dict[str, list[Verdict]], pass_overall: float = 3.5) -> dict:
    """Declare a winner between two configs (prompt v1 vs v2, or model A vs B)
    from pointwise verdicts over the same suite."""
    names = sorted(verdicts_by_config)
    if len(names) != 2:
        raise ValueError("compare_configs expects exactly two configs")
    stats = {n: pointwise_report(verdicts_by_config[n], pass_overall) for n in names}
    a, b = names
    winner = None
    if stats[a]["mean_overall"] is not None and stats[b]["mean_overall"] is not None:
        if stats[a]["mean_overall"] > stats[b]["mean_overall"]:
            winner = a
        elif stats[b]["mean_overall"] > stats[a]["mean_overall"]:
            winner = b
        else:
            winner = "tie"
    return {"configs": stats, "winner": winner,
            "decided_by": f"mean_overall over identical suite (pass_overall={pass_overall})"}
