"""Blind human annotation files and judge-agreement calibration."""

import hashlib
import json
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

    sample_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    evidence: list[str]
    model: str
    config_id: str
    judge_score: int = Field(ge=1, le=5)


class BlindAnnotation(BaseModel):
    """Sample shown to a human without model, config, or judge identity."""

    sample_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    evidence: list[str]
    content_hash: str = Field(min_length=64, max_length=64)
    human_score: int | None = Field(default=None, ge=1, le=5)


class HumanAnnotation(BaseModel):
    """Imported human score keyed by stable case ID."""

    sample_id: str
    human_score: int = Field(ge=1, le=5)
    content_hash: str = Field(min_length=64, max_length=64)


class AnnotationSnapshot(BaseModel):
    """Private mapping from one opaque sample to its scored source."""

    sample_id: str
    source_case_id: str
    config_id: str
    model: str
    judge_score: int = Field(ge=1, le=5)
    content_hash: str = Field(min_length=64, max_length=64)


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
) -> list[BlindAnnotation]:
    """Write JSONL with judge and system identity fields physically removed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotations = [
        BlindAnnotation(
            sample_id=sample.sample_id,
            question=sample.question,
            reference_answer=sample.reference_answer,
            candidate_answer=sample.candidate_answer,
            evidence=sample.evidence,
            content_hash=_content_hash(
                sample_id=sample.sample_id,
                question=sample.question,
                reference_answer=sample.reference_answer,
                candidate_answer=sample.candidate_answer,
                evidence=sample.evidence,
            ),
        ).model_dump_json()
        for sample in samples
    ]
    output_path.write_text("\n".join(annotations) + "\n", encoding="utf-8")
    return [BlindAnnotation.model_validate_json(line) for line in annotations]


def import_human_annotations(path: str | Path) -> list[HumanAnnotation]:
    """Validate completed blind JSONL and reject missing or duplicate labels."""

    annotations: list[HumanAnnotation] = []
    seen_sample_ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        blind = BlindAnnotation.model_validate_json(line)
        if blind.human_score is None:
            raise ValueError(f"human_score is required for sample {blind.sample_id}")
        expected_hash = _content_hash(
            sample_id=blind.sample_id,
            question=blind.question,
            reference_answer=blind.reference_answer,
            candidate_answer=blind.candidate_answer,
            evidence=blind.evidence,
        )
        if blind.content_hash != expected_hash:
            raise ValueError(f"content hash mismatch for sample {blind.sample_id}")
        if blind.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate human annotation: {blind.sample_id}")
        seen_sample_ids.add(blind.sample_id)
        annotations.append(
            HumanAnnotation(
                sample_id=blind.sample_id,
                human_score=blind.human_score,
                content_hash=blind.content_hash,
            )
        )
    return annotations


def _content_hash(
    *,
    sample_id: str,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    evidence: list[str],
) -> str:
    canonical = json.dumps(
        {
            "candidate_answer": candidate_answer,
            "evidence": evidence,
            "question": question,
            "reference_answer": reference_answer,
            "sample_id": sample_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
