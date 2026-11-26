# Problem 2 - LLM-as-Judge Evaluation System

A bias-checked evaluation harness that scores model outputs against a rubric,
compares two configurations (prompt versions or models) pairwise, and audits
its own judge reliability before you trust the numbers.

Built so a human can read every decision: the full judge log is kept as
JSONL, and every claim in the report traces back to a logged call.

## What it does

| Mode | Command | Output |
| --- | --- | --- |
| Pointwise scoring | `python src/run.py --suite suites/example_pointwise.json --mode pointwise` | per-criterion 1-5 scores + rationale per case |
| Pairwise A/B | `python src/run.py --suite suites/example_ab.json --mode pairwise` | winner per pair, both presentation orders |
| Judge validation | `python src/run.py --suite suites/probes.json --mode validate` | test-retest, position-flip, adversarial, gold-agreement report |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY
```

One key powers both the judged models and the judge itself. The judge model
defaults to a different family from the generation model so the judge is not
grading its own prose.

## Rubric

Six criteria, each scored 1-5 with written anchors (see `src/schema.py`):
correctness, completeness, instruction following, clarity, tone, and safety.
Anchors matter: "3" means something specific, not "middling", which keeps
scores comparable across cases and across runs.

## Bias handling

LLM judges have known failure modes. Each one has a concrete countermeasure
here, and `validate` mode measures whether the countermeasure works:

1. **Position bias** - judges favor whichever answer appears first.
   Every pairwise case runs in BOTH orders (A-then-B and B-then-A) and the
   winner is normalized. `position_flip_rate` in the report is the fraction
   of pairs where swapping order changed the verdict; high values mean the
   judge is reading position, not content.
2. **Verbosity bias** - longer answers score higher regardless of quality.
   `probes.json` contains a verbose-but-wrong vs terse-but-correct pair;
   `validate` reports whether the judge still picks the correct one.
3. **Self-preference** - a model judging its own family's output grades
   leniently. The judge model is configurable and defaults to a different
   provider family than the generation model.
4. **Inconsistency** - same input, different verdict on rerun.
   `test_retest` re-scores each case and reports agreement.
5. **No ground truth check** - an unaudited judge is an assumption.
   `gold_agreement` scores cases with known-correct and known-wrong outputs
   and reports how often the judge's verdict matches the gold label.

## Score aggregation

`src/aggregate.py` computes:
- pass rate (mean overall >= 4.0) and per-criterion means;
- for pairwise runs: win counts per config with position flips excluded
  from the tally (a flipped pair counts as "no reliable signal", not as a
  win for either side);
- the `compare_configs` winner statement used in A/B reports.

## Audit trail

Every judge call (prompt, raw response, parsed verdict, latency, retries)
is appended to `judge_log.jsonl` next to the output report. Malformed judge
JSON gets one repair retry; if it still fails, the case is recorded as a
parse failure and excluded from aggregates - never silently rescored or
dropped.

## Files

```
src/schema.py     dataclasses + rubric definition + robust JSON extraction
src/llm.py        OpenAI-compatible client (OpenRouter), 429 backoff
src/judge.py      pointwise + pairwise judging, both-orders, logging
src/aggregate.py  pass rate, means, flip rate, config comparison
src/validate.py   test-retest, adversarial probes, gold agreement
src/run.py        CLI entry point
suites/           example pointwise, A/B, and probe suites
```
