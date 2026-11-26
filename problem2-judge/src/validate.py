"""Judge validation: test-retest consistency + adversarial probes + gold agreement.

At least one validation is required by the assignment; this implements all three.
"""
from __future__ import annotations

from .judge import Judge
from .schema import TestCase


def test_retest(judge: Judge, cases: list[TestCase], runs: int = 2) -> dict:
    """Judge each case `runs` times; report how often the verdict flips."""
    flips = 0
    total = 0
    for case in cases:
        verdicts = [judge.judge_pointwise(case) for _ in range(runs)]
        overs = [v.overall for v in verdicts if v.parse_ok and v.overall is not None]
        if len(overs) == runs:
            total += 1
            if len(set(overs)) > 1:
                flips += 1
    return {"runs_per_case": runs, "n_cases": total,
            "verdict_flip_rate": round(flips / total, 3) if total else None}


def adversarial_probes(judge: Judge, probes: list[TestCase]) -> dict:
    """Probes: verbose-but-wrong (should score LOW) and terse-but-correct
    (should score HIGH). 'Fooled' = padded-wrong scores >= terse-correct."""
    results = []
    fooled = 0
    by_kind: dict[str, list[TestCase]] = {}
    for p in probes:
        kind = (p.criteria or ["unknown"])[0]
        by_kind.setdefault(kind, []).append(p)
    pairs = zip(by_kind.get("verbose_wrong", []), by_kind.get("terse_correct", []))
    for wrong, right in pairs:
        vw = judge.judge_pointwise(wrong)
        tc = judge.judge_pointwise(right)
        pair_fooled = (vw.parse_ok and tc.parse_ok and vw.overall is not None
                       and tc.overall is not None and vw.overall >= tc.overall)
        fooled += 1 if pair_fooled else 0
        results.append({
            "verbose_wrong_id": wrong.id, "verbose_wrong_overall": vw.overall,
            "terse_correct_id": right.id, "terse_correct_overall": tc.overall,
            "judge_fooled": pair_fooled,
        })
    return {"n_probe_pairs": len(results), "times_fooled": fooled, "detail": results}


def gold_agreement(judge: Judge, cases: list[TestCase], pass_overall: float = 3.5) -> dict:
    """Cases carry expected_output labelled pass/fail in criteria[0] ('gold_pass'/'gold_fail').
    Agreement = judge's pass/fail decision matching the gold label."""
    agree = 0
    n = 0
    for case in cases:
        gold_label = (case.criteria or [None])[0]
        if gold_label not in {"gold_pass", "gold_fail"}:
            continue
        v = judge.judge_pointwise(case)
        if not v.parse_ok or v.overall is None:
            continue
        n += 1
        judged_pass = v.overall >= pass_overall
        if judged_pass == (gold_label == "gold_pass"):
            agree += 1
    return {"n_labelled_cases": n,
            "agreement_rate": round(agree / n, 3) if n else None}
