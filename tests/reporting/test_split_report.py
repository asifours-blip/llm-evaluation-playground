"""Reports keep dev and holdout apart and mark missing labels."""

import json
from pathlib import Path
from typing import Any

from rag_quality_lab.domain.models import (
    UNLABELED,
    CaseResult,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
)
from rag_quality_lab.reporting.report import generate_reports


def result(case_id: str, split: str | None, recall: float, **extra: Any) -> CaseResult:
    return CaseResult.model_validate(
        {
            "case_id": case_id,
            "config_id": "config-a",
            "model": "fake-chat",
            "metrics": {"retrieval_recall_at_k": recall},
            "answerability": "answerable",
            "split": split,
            "status": "completed",
            **extra,
        }
    )


def experiment(holdout_freeze_hash: str | None) -> ExperimentRecord:
    return ExperimentRecord(
        id="split-report",
        identity=ExperimentIdentity.model_validate(
            {
                "name": "split",
                "mode": "mock",
                "commit_sha": "abc",
                "dirty": False,
                "dataset_hash": "dataset",
                "dataset_version": "1.0.0",
                "holdout_freeze_hash": holdout_freeze_hash,
                "prompt_hashes": {},
                "config": {},
                "random_seed": 42,
                "python_version": "3.11",
            }
        ),
        status=ExperimentStatus.COMPLETED,
        case_results=[
            result("rag-001", "dev", 1.0, difficulty="easy", review_status="approved"),
            result("rag-002", "dev", 0.0),
            result("rag-003", "holdout", 0.5, difficulty="hard"),
            result("rag-004", None, 0.25),
        ],
        summary={},
    )


def test_report_separates_dev_holdout_and_unlabeled_results(tmp_path: Path) -> None:
    paths = generate_reports(experiment("frozen-hash"), tmp_path)
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    splits = payload["split_breakdown"]

    assert list(splits) == ["dev", "holdout", UNLABELED]
    assert splits["dev"]["held_out"] is False
    assert splits["dev"]["metrics"]["config-a"]["retrieval_recall_at_k"] == 0.5
    assert splits["holdout"]["held_out"] is True
    assert splits["holdout"]["holdout_freeze_hash"] == "frozen-hash"
    assert splits["holdout"]["metrics"]["config-a"]["retrieval_recall_at_k"] == 0.5
    assert splits[UNLABELED]["held_out"] is False
    assert splits[UNLABELED]["case_count"] == 1
    labels = payload["dataset_labels"]
    assert labels["split"] == {"dev": 2, "holdout": 1, UNLABELED: 1}
    assert labels["difficulty"] == {"easy": 1, "hard": 1, UNLABELED: 2}
    assert labels["review_status"] == {"approved": 1, UNLABELED: 3}
    html = paths.html.read_text(encoding="utf-8")
    assert "not held-out evidence" in html
    assert UNLABELED in html


def test_unfrozen_holdout_is_not_presented_as_held_out(tmp_path: Path) -> None:
    paths = generate_reports(experiment(None), tmp_path)

    splits = json.loads(paths.json.read_text(encoding="utf-8"))["split_breakdown"]
    assert splits["holdout"]["held_out"] is False
    assert "not frozen" in splits["holdout"]["note"]
