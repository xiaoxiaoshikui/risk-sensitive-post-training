"""
Verifiable rewards.

Rewards are programmatically checkable rather than model-judged. That is a
deliberate scope choice: a learned or LLM judge introduces its own tail
failures, and this repository is trying to measure tail behaviour induced by
the *optimizer*, not by the reward model.

Correctness and format compliance are returned separately. Collapsing them
into one scalar makes a policy that degenerates into malformed output
indistinguishable from one that is merely wrong, and those need different
responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["RewardBreakdown", "gsm8k_reward", "extract_answer", "shaped_reward"]

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_GOLD_TAIL = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class RewardBreakdown:
    correct: bool
    format_ok: bool
    reward: float

    @property
    def as_tuple(self) -> tuple[bool, bool, float]:
        return (self.correct, self.format_ok, self.reward)


def _normalize_number(text: str) -> str | None:
    m = _NUMBER.search(text.replace(",", ""))
    if m is None:
        return None
    val = m.group(0)
    try:
        f = float(val)
    except ValueError:
        return None
    # Canonicalise so "42", "42.0" and "42.00" compare equal.
    return str(int(f)) if f == int(f) else repr(f)


def extract_answer(completion: str) -> tuple[str | None, bool]:
    """Return (normalized answer, format_ok).

    format_ok is True only when the completion contains exactly one
    well-formed <answer> block. Falling back to 'last number in the string'
    when the tag is missing would reward a policy for abandoning the output
    contract, which is precisely the degenerate behaviour worth measuring.
    """
    matches = _ANSWER_TAG.findall(completion)
    if len(matches) != 1:
        return (None, False)
    return (_normalize_number(matches[0]), True)


def gsm8k_gold(answer_field: str) -> str | None:
    """Parse the reference answer from a GSM8K-style '#### N' field."""
    m = _GOLD_TAIL.search(answer_field)
    if m is None:
        return _normalize_number(answer_field)
    return _normalize_number(m.group(1))


def gsm8k_reward(
    completion: str,
    answer_field: str,
    *,
    format_bonus: float = 0.1,
) -> RewardBreakdown:
    """Binary correctness plus a small format bonus.

    The format bonus is kept an order of magnitude below the correctness
    signal so it shapes early exploration without becoming worth gaming once
    the policy can actually solve prompts.
    """
    gold = gsm8k_gold(answer_field)
    pred, format_ok = extract_answer(completion)
    correct = bool(gold is not None and pred is not None and pred == gold)
    reward = (1.0 if correct else 0.0) + (format_bonus if format_ok else 0.0)
    return RewardBreakdown(correct=correct, format_ok=format_ok, reward=reward)


def shaped_reward(breakdown: RewardBreakdown, *, length: int, max_length: int,
                  length_penalty: float = 0.0) -> float:
    """Optional length penalty, off by default.

    Included because verbosity drift is a common post-training artifact, but
    left disabled: adding it by default would confound the risk-estimator
    comparison this repository is built to run.
    """
    r = breakdown.reward
    if length_penalty > 0.0 and length > max_length:
        r -= length_penalty * (length - max_length) / max(max_length, 1)
    return r
