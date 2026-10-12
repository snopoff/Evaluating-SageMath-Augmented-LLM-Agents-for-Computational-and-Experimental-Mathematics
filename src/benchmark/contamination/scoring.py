"""Overlap metrics between a model continuation and the true continuation.

All metrics work on a crude word/symbol tokenisation of the text, which is enough for
verbatim-recall comparisons and keeps the module dependency-free:

* ``rouge_l``          - LCS-based F1 (Lin, 2004), the usual continuation score.
* ``ngram_hit_rate``   - fraction of the model's 8-grams that occur in the target: the
                         8-gram threshold is the one used for training-set decontamination.
* ``longest_run``      - longest verbatim run in tokens; 50 tokens is the threshold used
                         for "extractable memorisation" (Carlini et al., 2023).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
LETTER_RE = re.compile(r"\b([A-E])\b")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, target: str) -> float:
    pred, ref = tokenize(prediction), tokenize(target)
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def ngram_hit_rate(prediction: str, target: str, n: int = 8) -> float:
    pred, ref = tokenize(prediction), tokenize(target)
    if len(pred) < n or len(ref) < n:
        return 0.0
    ref_grams = {tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)}
    pred_grams = [tuple(pred[i : i + n]) for i in range(len(pred) - n + 1)]
    return sum(1 for gram in pred_grams if gram in ref_grams) / len(pred_grams)


def longest_run(prediction: str, target: str) -> int:
    pred, ref = tokenize(prediction), tokenize(target)
    if not pred or not ref:
        return 0
    best = 0
    previous = [0] * (len(ref) + 1)
    for token in pred:
        current = [0] * (len(ref) + 1)
        for j, other in enumerate(ref, start=1):
            if token == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


@dataclass(frozen=True)
class ContinuationScore:
    rouge_l: float
    ngram8_hit: float
    longest_run: int
    prediction_tokens: int
    target_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rouge_l": round(self.rouge_l, 4),
            "ngram8_hit": round(self.ngram8_hit, 4),
            "longest_run": self.longest_run,
            "prediction_tokens": self.prediction_tokens,
            "target_tokens": self.target_tokens,
        }


def score_continuation(prediction: str, target: str) -> ContinuationScore:
    return ContinuationScore(
        rouge_l=rouge_l(prediction, target),
        ngram8_hit=ngram_hit_rate(prediction, target),
        longest_run=longest_run(prediction, target),
        prediction_tokens=len(tokenize(prediction)),
        target_tokens=len(tokenize(target)),
    )


def parse_quiz_letter(text: str) -> str | None:
    match = LETTER_RE.search((text or "").strip().upper())
    return match.group(1) if match else None


REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i'm not able",
    "i am not able",
    "i don't have",
    "i do not have",
    "cannot provide",
    "can't provide",
    "cannot reproduce",
    "can't reproduce",
    "not able to reproduce",
    "i'm unable",
    "i am unable",
    "as an ai",
)


def looks_like_refusal(text: str) -> bool:
    """A reply that talks about itself instead of continuing the passage."""

    head = (text or "")[:250].lower()
    return any(marker in head for marker in REFUSAL_MARKERS)


def parse_instant_answer(text: str) -> dict[str, Any]:
    """Extract {final_answer, sympy_answer} from an instant-answer reply (JSON or free text)."""

    raw = (text or "").strip()
    candidate = raw
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return {
                "final_answer": str(payload.get("final_answer", "")).strip(),
                "sympy_answer": payload.get("sympy_answer", ""),
                "parsed": True,
            }
    except json.JSONDecodeError:
        pass
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        try:
            payload = json.loads(brace.group(0))
            if isinstance(payload, dict):
                return {
                    "final_answer": str(payload.get("final_answer", "")).strip(),
                    "sympy_answer": payload.get("sympy_answer", ""),
                    "parsed": True,
                }
        except json.JSONDecodeError:
            pass
    return {"final_answer": raw, "sympy_answer": "", "parsed": False}
