"""Blind human annotation files and judge-agreement calibration."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field


class HumanJudgePair(BaseModel):
    """One aligned scalar score from a human and the model judge."""

    human_score: int = Field(ge=1, le=5)
    judge_score: int = Field(ge=1, le=5)


class CalibrationResult(BaseModel):
    """Agreement evidence controlling whether judge metrics may block."""

    label_count: int
    exact_agreement: float
    within_one_rate: float
    mean_absolute_error: float
    blocking_eligible: bool
    reason: str


class JudgeSample(BaseModel):
    """Complete scored sample before identity fields are blinded."""

    case_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    evidence: list[str]
    model: str
    config_id: str
    judge_score: int = Field(ge=1, le=5)


class BlindAnnotation(BaseModel):
    """Sample shown to a human without model, config, or judge identity."""

    case_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    evidence: list[str]
    human_score: int | None = Field(default=None, ge=1, le=5)


class HumanAnnotation(BaseModel):
    """Imported human score keyed by stable case ID."""

    case_id: str
    human_score: int = Field(ge=1, le=5)


def calibrate(pairs: Sequence[HumanJudgePair]) -> CalibrationResult:
    """Measure judge agreement and enforce the blocking-eligibility thresholds."""

    count = len(pairs)
    if count == 0:
        exact_agreement = 0.0
        within_one_rate = 0.0
        mean_absolute_error = 0.0
    else:
        absolute_errors = [
            abs(pair.human_score - pair.judge_score) for pair in pairs
        ]
        exact_agreement = sum(error == 0 for error in absolute_errors) / count
        within_one_rate = sum(error <= 1 for error in absolute_errors) / count
        mean_absolute_error = sum(absolute_errors) / count

    if count < 12:
        blocking_eligible = False
        reason = "at least 12 labels are required"
    elif within_one_rate >= 0.8 and mean_absolute_error <= 1.0:
        blocking_eligible = True
        reason = "agreement thresholds met"
    else:
        blocking_eligible = False
        reason = "agreement thresholds not met"
    return CalibrationResult(
        label_count=count,
        exact_agreement=exact_agreement,
        within_one_rate=within_one_rate,
        mean_absolute_error=mean_absolute_error,
        blocking_eligible=blocking_eligible,
        reason=reason,
    )


def export_blind_annotations(
    samples: Sequence[JudgeSample], path: str | Path
) -> None:
    """Write JSONL with judge and system identity fields physically removed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        BlindAnnotation(
            case_id=sample.case_id,
            question=sample.question,
            reference_answer=sample.reference_answer,
            candidate_answer=sample.candidate_answer,
            evidence=sample.evidence,
        ).model_dump_json()
        for sample in samples
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_human_annotations(path: str | Path) -> list[HumanAnnotation]:
    """Validate completed blind JSONL and reject missing or duplicate labels."""

    annotations: list[HumanAnnotation] = []
    seen_case_ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        blind = BlindAnnotation.model_validate_json(line)
        if blind.human_score is None:
            raise ValueError(f"human_score is required for case {blind.case_id}")
        if blind.case_id in seen_case_ids:
            raise ValueError(f"duplicate human annotation: {blind.case_id}")
        seen_case_ids.add(blind.case_id)
        annotations.append(
            HumanAnnotation(case_id=blind.case_id, human_score=blind.human_score)
        )
    return annotations
