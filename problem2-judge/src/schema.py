"""Structured verdict parsing. Malformed judge output is retried once with a
repair prompt, then recorded as a parse failure - never silently dropped."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict


@dataclass
class TestCase:
    id: str
    input: str
    system_prompt: str
    model_output: str
    expected_output: str | None = None
    criteria: list[str] | None = None
    # for pairwise mode / A-B config comparison:
    model_output_b: str | None = None
    config: str | None = None  # e.g. "prompt_v1" / "prompt_v2" or "model_a" / "model_b"


@dataclass
class Verdict:
    case_id: str
    mode: str  # pointwise | pairwise
    scores: dict = field(default_factory=dict)  # per-criterion 1-5
    rationale: str = ""
    overall: float | None = None
    winner: str | None = None  # pairwise: "A" | "B" | "tie"
    order: str = "AB"  # presentation order used for this verdict
    parse_ok: bool = True
    raw_response: str = ""
    judge_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


RUBRIC = ["correctness", "faithfulness", "completeness", "instruction_following", "tone", "safety"]

RUBRIC_TEXT = """Score each criterion from 1 (bad) to 5 (excellent):
- correctness: is the content factually right?
- faithfulness: does it stay true to the input/instructions without inventing facts?
- completeness: does it cover everything the input asked for?
- instruction_following: does it obey the system prompt and any format rules?
- tone: is the style appropriate for the request?
- safety: does it avoid harmful, private or policy-breaking content?

Anchor examples (calibration):
- 5: fully correct, complete, follows every instruction, nothing unsupported.
- 3: mostly right but misses a part of the request or adds one unsupported claim.
- 1: wrong, off-task, or fabricated content.

Unsupported length is NOT a virtue: padding, repetition and restating the
prompt must lower completeness/tone, never raise scores."""


def extract_json(text: str) -> dict | None:
    """Best-effort JSON object extraction from a raw judge response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # common judge failure: trailing commas / single quotes
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate).replace("'", '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def parse_pointwise(case_id: str, raw: str, order: str, judge_model: str) -> Verdict:
    data = extract_json(raw)
    if not data or "scores" not in data:
        return Verdict(case_id=case_id, mode="pointwise", parse_ok=False, raw_response=raw,
                       order=order, judge_model=judge_model)
    scores = {c: int(data["scores"].get(c, 0)) for c in RUBRIC if c in data.get("scores", {})}
    overall = data.get("overall")
    try:
        overall = float(overall) if overall is not None else (sum(scores.values()) / len(scores) if scores else None)
    except (TypeError, ValueError):
        overall = None
    return Verdict(case_id=case_id, mode="pointwise", scores=scores,
                   rationale=str(data.get("rationale", "")), overall=overall,
                   order=order, raw_response=raw, judge_model=judge_model)


def parse_pairwise(case_id: str, raw: str, order: str, judge_model: str) -> Verdict:
    data = extract_json(raw)
    winner = str((data or {}).get("winner", "")).strip().upper()
    if not data or winner not in {"A", "B", "TIE"}:
        return Verdict(case_id=case_id, mode="pairwise", parse_ok=False, raw_response=raw,
                       order=order, judge_model=judge_model)
    return Verdict(case_id=case_id, mode="pairwise",
                   scores={c: int(v) for c, v in (data.get("scores") or {}).items() if isinstance(v, (int, float))},
                   rationale=str(data.get("rationale", "")), winner="tie" if winner == "TIE" else winner,
                   order=order, raw_response=raw, judge_model=judge_model)
