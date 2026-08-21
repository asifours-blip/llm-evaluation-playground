import pytest
from pydantic import ValidationError

from rag_quality_lab.domain.models import JudgeVerdict
from rag_quality_lab.metrics.judge import (
    PairwiseVerdict,
    build_pairwise_judge_prompt,
    build_scalar_judge_prompt,
    parse_judge_verdict,
    parse_pairwise_verdict,
    resolve_pairwise,
)


def test_judge_verdict_enforces_pass_threshold() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(score=3, passed=True, reason="inconsistent")


def test_parse_judge_verdict_validates_json_contract() -> None:
    verdict = parse_judge_verdict('{"score":4,"passed":true,"reason":"grounded"}')

    assert verdict.score == 4
    with pytest.raises(ValidationError):
        parse_judge_verdict('{"score":2,"passed":true,"reason":"wrong"}')


def test_scalar_prompt_contains_rubric_and_evidence() -> None:
    prompt = build_scalar_judge_prompt(
        question="Q",
        reference_answer="R",
        candidate_answer="A",
        evidence=["E"],
    )

    assert "1 to 5" in prompt
    assert "E" in prompt
    assert "score" in prompt


def test_scalar_prompt_caps_scores_when_core_reference_content_is_missing() -> None:
    prompt = build_scalar_judge_prompt(
        question="How should a report diagnose failures?",
        reference_answer="Report P50 and distinguish retrieval misses.",
        candidate_answer="Store the commit SHA.",
        evidence=["E"],
    )

    assert "Score 1" in prompt
    assert "Score 2" in prompt
    assert "missing the central reference requirement" in prompt
    assert "must not receive a score of 4 or 5" in prompt


def test_pairwise_resolution_normalizes_reversed_labels() -> None:
    result = resolve_pairwise(
        forward=PairwiseVerdict(preferred="A", reason="better"),
        reversed_order=PairwiseVerdict(preferred="B", reason="still better"),
    )

    assert result.winner == "A"
    assert not result.position_sensitive


def test_pairwise_resolution_excludes_position_sensitive_result() -> None:
    result = resolve_pairwise(
        forward=PairwiseVerdict(preferred="A", reason="first"),
        reversed_order=PairwiseVerdict(preferred="A", reason="first again"),
    )

    assert result.winner is None
    assert result.position_sensitive


def test_pairwise_prompt_and_tie_are_order_stable() -> None:
    prompt = build_pairwise_judge_prompt(
        question="Q",
        reference_answer="R",
        evidence=["E"],
        answer_a="first",
        answer_b="second",
    )
    result = resolve_pairwise(
        forward=PairwiseVerdict(preferred="tie", reason="equal"),
        reversed_order=PairwiseVerdict(preferred="tie", reason="equal"),
    )

    assert "A:\nfirst" in prompt
    assert "B:\nsecond" in prompt
    assert result.winner == "tie"
    assert not result.position_sensitive


def test_pairwise_parser_rejects_empty_reason() -> None:
    with pytest.raises(ValidationError):
        parse_pairwise_verdict('{"preferred":"A","reason":""}')
