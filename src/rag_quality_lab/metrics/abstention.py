"""Confusion-matrix metrics for answer-versus-abstain decisions."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AbstentionObservation:
    """Expected answerability and the system's observed abstention."""

    expected_answerable: bool
    abstained: bool


@dataclass(frozen=True)
class AbstentionSummary:
    """Aggregate metrics where the positive class is 'should abstain'."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    false_answer_rate: float
    over_abstention_rate: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


def is_effective_abstention(*, abstained: bool, answer: str) -> bool:
    """Require both the structured flag and an explicit non-answer in the text."""

    if not abstained:
        return False
    opening = " ".join(answer.casefold().split())[:180]
    refusal_markers = (
        "does not contain",
        "does not provide",
        "does not specify",
        "does not describe",
        "does not state",
        "contains no",
        "has no",
        "gives no",
        "defines no",
        "names no",
        "no evidence",
        "insufficient evidence",
        "cannot answer",
        "can't answer",
        "not enough information",
        "没有",
        "无法回答",
        "信息不足",
    )
    return any(marker in opening for marker in refusal_markers)


def summarize_abstention(
    observations: Sequence[AbstentionObservation],
) -> AbstentionSummary:
    true_positive = sum(
        not item.expected_answerable and item.abstained for item in observations
    )
    false_positive = sum(
        item.expected_answerable and item.abstained for item in observations
    )
    false_negative = sum(
        not item.expected_answerable and not item.abstained for item in observations
    )
    true_negative = sum(
        item.expected_answerable and not item.abstained for item in observations
    )

    accuracy = _safe_divide(true_positive + true_negative, len(observations))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    false_answer_rate = _safe_divide(false_negative, true_positive + false_negative)
    over_abstention_rate = _safe_divide(false_positive, false_positive + true_negative)
    return AbstentionSummary(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_answer_rate=false_answer_rate,
        over_abstention_rate=over_abstention_rate,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
