from rag_quality_lab.metrics.abstention import (
    AbstentionObservation,
    summarize_abstention,
)


def test_false_answer_and_over_abstention_rates() -> None:
    observations = [
        AbstentionObservation(expected_answerable=False, abstained=False),
        AbstentionObservation(expected_answerable=False, abstained=True),
        AbstentionObservation(expected_answerable=True, abstained=True),
        AbstentionObservation(expected_answerable=True, abstained=False),
    ]

    summary = summarize_abstention(observations)

    assert summary.accuracy == 0.5
    assert summary.precision == 0.5
    assert summary.recall == 0.5
    assert summary.f1 == 0.5
    assert summary.false_answer_rate == 0.5
    assert summary.over_abstention_rate == 0.5


def test_empty_abstention_summary_uses_zero_denominators() -> None:
    summary = summarize_abstention([])

    assert summary.accuracy == 0.0
    assert summary.f1 == 0.0
    assert summary.false_answer_rate == 0.0
