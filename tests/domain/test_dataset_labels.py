"""Dataset version, content hash, and per-case labels."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_quality_lab.config.loaders import load_dataset
from rag_quality_lab.domain.models import UNLABELED, CaseReview, EvaluationDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def legacy_case(case_id: str, **overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": case_id,
        "question": f"Question {case_id}?",
        "reference_answer": "Answer.",
        "answerability": "answerable",
        "expected_document_ids": ["doc-01"],
        "reference_evidence": ["Evidence."],
        "category": "retrieval",
        "difficulty": "easy",
    }
    case.update(overrides)
    return case


def write_dataset(path: Path, cases: list[dict[str, Any]], version: str = "1.0.0") -> Path:
    path.write_text(
        json.dumps({"version": version, "name": "labels", "cases": cases}),
        encoding="utf-8",
    )
    return path


def test_committed_legacy_dataset_loads_with_unlabeled_defaults_and_same_hash() -> None:
    dataset = load_dataset(REPOSITORY_ROOT / "data/eval/rag_quality_v1.json")
    archived = json.loads(
        (REPOSITORY_ROOT / "docs/artifacts/offline-summary.json").read_text(encoding="utf-8")
    )

    assert all(case.split is None for case in dataset.cases)
    assert all(case.review is None for case in dataset.cases)
    assert dataset.version == "1.0.0"
    # The content hash is the frozen dataset_hash of the archived offline evidence.
    assert dataset.content_hash() == archived["identity"]["dataset_hash"]


def test_missing_difficulty_split_and_review_default_to_unlabeled(tmp_path: Path) -> None:
    case = legacy_case("rag-001")
    del case["difficulty"]

    dataset = load_dataset(write_dataset(tmp_path / "old.json", [case]))

    loaded = dataset.cases[0]
    assert loaded.difficulty is None
    assert loaded.split is None
    assert loaded.review is None
    assert loaded.is_unanswerable is False
    assert loaded.label_values() == {
        "difficulty": UNLABELED,
        "answerability": "answerable",
        "review_status": UNLABELED,
        "split": UNLABELED,
    }


def test_new_labels_round_trip_through_the_dataset_file(tmp_path: Path) -> None:
    cases = [
        legacy_case(
            "rag-001",
            difficulty="hard",
            split="dev",
            review={
                "status": "approved",
                "reviewer": "reviewer-a",
                "reviewed_at": "2026-09-01T08:00:00Z",
            },
        ),
        legacy_case(
            "rag-002",
            answerability="unanswerable",
            expected_document_ids=[],
            reference_evidence=[],
            split="holdout",
            review={"status": "unreviewed"},
        ),
    ]
    path = write_dataset(tmp_path / "labelled.json", cases, version="2.0.0")

    dataset = load_dataset(path)
    path.write_text(dataset.model_dump_json(exclude_none=True), encoding="utf-8")
    reloaded = load_dataset(path)

    assert reloaded == dataset
    first, second = reloaded.cases
    assert first.split == "dev"
    assert first.review == CaseReview.model_validate(
        {
            "status": "approved",
            "reviewer": "reviewer-a",
            "reviewed_at": "2026-09-01T08:00:00Z",
        }
    )
    assert second.is_unanswerable is True
    assert second.label_values()["review_status"] == "unreviewed"
    assert reloaded.split_case_ids("holdout") == ["rag-002"]
    assert reloaded.content_hash() == dataset.content_hash()


@pytest.mark.parametrize("status", ["approved", "rejected"])
def test_decided_review_requires_reviewer_and_time(status: str) -> None:
    with pytest.raises(ValidationError, match="reviewer"):
        CaseReview.model_validate({"status": status})
    with pytest.raises(ValidationError):
        CaseReview.model_validate({"status": "maybe"})


def test_content_hash_changes_with_any_label(tmp_path: Path) -> None:
    base = load_dataset(write_dataset(tmp_path / "a.json", [legacy_case("rag-001")]))
    labelled = load_dataset(
        write_dataset(tmp_path / "b.json", [legacy_case("rag-001", split="dev")])
    )

    assert base.content_hash() != labelled.content_hash()


def test_select_splits_keeps_only_requested_cases() -> None:
    dataset = EvaluationDataset.model_validate(
        {
            "version": "1.0.0",
            "name": "labels",
            "cases": [
                legacy_case("rag-001", split="dev"),
                legacy_case("rag-002", split="holdout"),
                legacy_case("rag-003"),
            ],
        }
    )

    assert [case.id for case in dataset.select_splits(["dev"]).cases] == ["rag-001"]
    assert dataset.select_splits(None) is dataset
    with pytest.raises(ValueError, match="no cases"):
        EvaluationDataset.model_validate(
            {"version": "1", "name": "x", "cases": [legacy_case("rag-001")]}
        ).select_splits(["holdout"])
