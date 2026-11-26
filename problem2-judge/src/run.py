"""CLI: suite in -> report out.

Usage:
  python -m src.run --suite suites/example_pointwise.json --mode pointwise
  python -m src.run --suite suites/example_ab.json --mode pairwise
  python -m src.run --suite suites/probes.json --mode validate

Config via env (see .env.example). Every judge call is logged to judge_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .aggregate import compare_configs, pairwise_report, pointwise_report
from .judge import Judge
from .llm import LLMClient
from .schema import TestCase
from .validate import adversarial_probes, gold_agreement, test_retest

load_dotenv()


def load_suite(path: str) -> list[TestCase]:
    raw = yaml.safe_load(Path(path).read_text())
    cases = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    return [TestCase(**c) for c in cases]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--mode", choices=["pointwise", "pairwise", "validate"], required=True)
    ap.add_argument("--out", default="report.json")
    args = ap.parse_args()

    judge_model = os.environ.get("JUDGE_MODEL", "google/gemma-4-26b-a4b-it:free")
    generator_model = os.environ.get("GENERATOR_MODEL", "nvidia/nemotron-3.5-lightning:free")
    llm = LLMClient()
    judge = Judge(llm, judge_model)
    cases = load_suite(args.suite)

    if args.mode == "pointwise":
        verdicts = [judge.judge_pointwise(c) for c in cases]
        report = {"mode": "pointwise", "judge_model": judge_model,
                  "generator_model": generator_model,
                  "report": pointwise_report(verdicts),
                  "verdicts": [v.to_dict() for v in verdicts]}
    elif args.mode == "pairwise":
        pairs = [judge.judge_pairwise(c) for c in cases]
        report = {"mode": "pairwise", "judge_model": judge_model,
                  "generator_model": generator_model,
                  "report": pairwise_report(pairs),
                  "verdicts": [[v.to_dict() for v in pair] for pair in pairs]}
    else:
        probes = [c for c in cases if (c.criteria or [""])[0] in {"verbose_wrong", "terse_correct"}]
        gold = [c for c in cases if (c.criteria or [""])[0] in {"gold_pass", "gold_fail"}]
        base = [c for c in cases if (c.criteria or [""])[0] not in
                {"verbose_wrong", "terse_correct", "gold_pass", "gold_fail"}]
        report = {"mode": "validate", "judge_model": judge_model,
                  "test_retest": test_retest(judge, base or cases[:3]),
                  "adversarial_probes": adversarial_probes(judge, probes) if probes else "no probes in suite",
                  "gold_agreement": gold_agreement(judge, gold) if gold else "no gold-labelled cases in suite"}

    report["judge_usage"] = {"calls": judge.calls, "tokens": judge.tokens}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:3000])


if __name__ == "__main__":
    main()
