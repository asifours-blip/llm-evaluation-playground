"""Structured scalar and order-controlled pairwise judge helpers."""

import json
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from rag_quality_lab.domain.models import JudgeVerdict, PairwiseVerdict, TokenUsage

Preference = Literal["A", "B", "tie"]


class PairwiseResult(BaseModel):
    """Preference normalized to the original candidate identities."""

    winner: Preference | None
    position_sensitive: bool
    forward: PairwiseVerdict
    reversed_order: PairwiseVerdict


class PairwiseCaseResult(BaseModel):
    """Persisted two-order comparison for one case and model."""

    case_id: str
    model: str
    status: Literal["completed", "failed"]
    winner: Literal["baseline", "candidate", "tie"] | None = None
    position_sensitive: bool = False
    forward: PairwiseVerdict | None = None
    reversed_order: PairwiseVerdict | None = None
    forward_usage: TokenUsage | None = None
    reversed_usage: TokenUsage | None = None
    cost: Decimal = Decimal("0")
    cost_estimated: bool = False
    error: str | None = None


class PairwiseComparisonRecord(BaseModel):
    """Auditable result of comparing two configurations in both answer orders."""

    id: str
    baseline_experiment_id: str
    candidate_experiment_id: str
    baseline_config_id: str
    candidate_config_id: str
    judge_model: str
    outcomes: list[PairwiseCaseResult]
    summary: dict[str, float]


def parse_judge_verdict(content: str) -> JudgeVerdict:
    """Parse scalar judge JSON through the score/pass invariant."""

    return JudgeVerdict.model_validate(json.loads(content))


def parse_pairwise_verdict(content: str) -> PairwiseVerdict:
    """Parse a pairwise JSON preference through the typed contract."""

    return PairwiseVerdict.model_validate(json.loads(content))


def build_scalar_judge_prompt(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    evidence: list[str],
) -> str:
    """Build the fixed 1-to-5 correctness and faithfulness rubric."""

    evidence_text = "\n".join(evidence)
    return (
        "Score the candidate from 1 to 5 for correctness and faithfulness using "
        "only the reference and evidence. Score 5 only when it answers the question "
        "and covers every core reference requirement. Score 4 only when it is mostly "
        "correct with a minor omission. Score 3 for a partially correct answer with a "
        "material omission. Score 2 when it is relevant but missing the central reference "
        "requirement. Score 1 when it is wrong, unsupported, or refuses an answerable "
        "question. A candidate missing the central reference requirement must not receive "
        "a score of 4 or 5. Scores 4 and 5 pass. Return JSON with keys \"score\", "
        '"passed", and "reason".\n\n'
        f"Question:\n{question}\n\nReference:\n{reference_answer}\n\n"
        f"Evidence:\n{evidence_text}\n\nCandidate:\n{candidate_answer}"
    )


def build_pairwise_judge_prompt(
    *,
    question: str,
    reference_answer: str,
    evidence: list[str],
    answer_a: str,
    answer_b: str,
) -> str:
    """Build a pairwise rubric whose answer order is controlled by the caller."""

    evidence_text = "\n".join(evidence)
    return (
        "Choose A, B, or tie using correctness and faithfulness only. Return JSON "
        'with keys "preferred" and "reason".\n\n'
        f"Question:\n{question}\n\nReference:\n{reference_answer}\n\n"
        f"Evidence:\n{evidence_text}\n\nA:\n{answer_a}\n\nB:\n{answer_b}"
    )


def resolve_pairwise(
    *,
    forward: PairwiseVerdict,
    reversed_order: PairwiseVerdict,
) -> PairwiseResult:
    """Normalize A/B and B/A verdicts and exclude order-sensitive preferences."""

    reversed_normalized = _reverse_preference(reversed_order.preferred)
    if forward.preferred == reversed_normalized:
        return PairwiseResult(
            winner=forward.preferred,
            position_sensitive=False,
            forward=forward,
            reversed_order=reversed_order,
        )
    return PairwiseResult(
        winner=None,
        position_sensitive=True,
        forward=forward,
        reversed_order=reversed_order,
    )


def _reverse_preference(preference: Preference) -> Preference:
    if preference == "A":
        return "B"
    if preference == "B":
        return "A"
    return "tie"
