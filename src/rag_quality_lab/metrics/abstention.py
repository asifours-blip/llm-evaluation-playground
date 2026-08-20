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
