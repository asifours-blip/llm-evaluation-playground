from rag_quality_lab.metrics.abstention import is_effective_abstention


def test_effective_abstention_requires_flag_and_refusal_text() -> None:
    assert is_effective_abstention(
        abstained=True,
        answer="The controlled corpus does not contain weather forecasts.",
    )
    assert not is_effective_abstention(
        abstained=True,
        answer="Tomorrow will be sunny and 25 degrees.",
    )
    assert not is_effective_abstention(
        abstained=False,
        answer="The corpus does not contain that information.",
    )
