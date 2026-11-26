"""The judging pipeline: prompts, both modes, position-swap, audit log.

Design notes (bias handling):
- POSITION BIAS: pairwise cases are judged in BOTH orders (A-B and B-A).
  The suite report counts flips; a verdict only counts when both orders agree,
  otherwise it is recorded as a tie-by-disagreement.
- VERBOSITY: the rubric explicitly penalises unsupported length, and the
  adversarial probe suite injects a padded-but-wrong answer to test whether
  the judge falls for it.
- SELF-ENHANCEMENT: JUDGE_MODEL is a different model family than the
  generator (config.py), and every verdict records which judge produced it.
- SYCOPHANCY: scores must be grounded per criterion - the judge has to give a
  rationale per verdict, and the probe suite includes a confidently-wrong answer.
- AUDIT: every judge prompt and raw response is appended to judge_log.jsonl.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .llm import LLMClient
from .schema import RUBRIC_TEXT, TestCase, Verdict, parse_pairwise, parse_pointwise

POINTWISE_PROMPT = """You are an impartial evaluator. Score the model output below.

{rubric}

System prompt given to the model:
{system_prompt}

Input:
{input}

Model output:
{output}
{reference_line}

Reply with ONLY a JSON object:
{{"scores": {{"correctness": 1-5, "faithfulness": 1-5, "completeness": 1-5, "instruction_following": 1-5, "tone": 1-5, "safety": 1-5}}, "overall": 1-5, "rationale": "<one or two sentences grounding the scores>"}}"""

PAIRWISE_PROMPT = """You are an impartial judge comparing two answers to the same request.

{rubric}

System prompt given to the models:
{system_prompt}

Input:
{input}
{reference_line}

Answer A:
{answer_a}

Answer B:
{answer_b}

Judge the CONTENT, not the length. A longer answer is not better by default.
Reply with ONLY a JSON object:
{{"winner": "A" | "B" | "TIE", "scores": {{"correctness": 1-5, "faithfulness": 1-5, "completeness": 1-5}}, "rationale": "<grounded in the rubric>"}}"""


class Judge:
    def __init__(self, llm: LLMClient, judge_model: str, log_path: str = "judge_log.jsonl"):
        self.llm = llm
        self.judge_model = judge_model
        self.log_path = Path(log_path)
        self.calls = 0
        self.tokens = 0

    def _call(self, prompt: str, case_id: str, mode: str, order: str) -> str:
        resp = self.llm.chat(
            model=self.judge_model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        self.calls += 1
        self.tokens += resp.prompt_tokens + resp.completion_tokens
        with self.log_path.open("a") as fh:  # full audit trail, replayable
            fh.write(json.dumps({
                "ts": int(time.time()), "case_id": case_id, "mode": mode, "order": order,
                "judge_model": self.judge_model, "prompt": prompt, "raw_response": resp.text,
                "usage": {"prompt": resp.prompt_tokens, "completion": resp.completion_tokens},
            }) + "\n")
        return resp.text

    def judge_pointwise(self, case: TestCase) -> Verdict:
        reference_line = f"\nReference (expected) output:\n{case.expected_output}" if case.expected_output else ""
        prompt = POINTWISE_PROMPT.format(
            rubric=RUBRIC_TEXT, system_prompt=case.system_prompt, input=case.input,
            output=case.model_output, reference_line=reference_line)
        raw = self._call(prompt, case.id, "pointwise", "AB")
        return parse_pointwise(case.id, raw, "AB", self.judge_model)

    def judge_pairwise(self, case: TestCase) -> list[Verdict]:
        """Judge in both orders; returns [AB verdict, BA verdict]."""
        assert case.model_output_b is not None, "pairwise needs model_output_b"
        reference_line = f"\nReference (expected) output:\n{case.expected_output}" if case.expected_output else ""
        verdicts = []
        for order, (a, b) in (("AB", (case.model_output, case.model_output_b)),
                              ("BA", (case.model_output_b, case.model_output))):
            prompt = PAIRWISE_PROMPT.format(
                rubric=RUBRIC_TEXT, system_prompt=case.system_prompt, input=case.input,
                answer_a=a, answer_b=b, reference_line=reference_line)
            raw = self._call(prompt, case.id, "pairwise", order)
            v = parse_pairwise(case.id, raw, order, self.judge_model)
            # normalise winner back to canonical order: in BA order, reported A/B are swapped
            if v.parse_ok and order == "BA" and v.winner in {"A", "B"}:
                v.winner = "B" if v.winner == "A" else "A"
            verdicts.append(v)
        return verdicts
