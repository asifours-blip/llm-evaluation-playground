"""Frozen holdout splits: tampering is refused, a new version is allowed."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_quality_lab.config.holdout import (
    HoldoutTampered,
    freeze_holdout,
    holdout_lock_path,
    load_holdout_lock,
    split_summary,
    verify_holdout,
)
from rag_quality_lab.config.loaders import load_dataset, load_experiment_config
from rag_quality_lab.domain.models import ExperimentRecord, ExperimentStatus, StructuredAnswer
from rag_quality_lab.experiments.runner import ProviderBundle, run_experiment
from rag_quality_lab.providers.fake import FakeChatProvider, FakeEmbeddingProvider


def case(case_id: str, split: str | None, question: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": case_id,
        "question": question or f"What does RAG retrieve for {case_id}?",
        "reference_answer": "RAG retrieves evidence.",
        "answerability": "answerable",
        "expected_document_ids": ["doc-01"],
        "reference_evidence": ["RAG retrieves evidence."],
        "category": "retrieval",
        "difficulty": "easy",
    }
    if split is not None:
        payload["split"] = split
    return payload


def write_dataset(path: Path, cases: list[dict[str, Any]], version: str = "1.0.0") -> Path:
    path.write_text(
        json.dumps({"version": version, "name": "holdout-demo", "cases": cases}),
        encoding="utf-8",
    )
    return path


def base_cases() -> list[dict[str, Any]]:
    return [case("rag-001", "dev"), case("rag-002", "dev"), case("rag-003", "holdout")]


def write_config(tmp_path: Path, dataset_path: Path) -> Path:
    corpus = tmp_path / "knowledge_base"
    corpus.mkdir(exist_ok=True)
    (corpus / "doc-01-rag.md").write_text(
        "# RAG\n\nID: doc-01\n\nRAG retrieves evidence.", encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "holdout",
                "mode": "mock",
                "dataset_path": str(dataset_path),
                "knowledge_base_path": str(corpus),
                "database_path": str(tmp_path / "runs.sqlite3"),
                "artifact_dir": str(tmp_path / "reports"),
                "max_workers": 1,
                "provider": {
                    "name": "fake",
                    "base_url": "https://offline.invalid/v1",
                    "api_key_env": "UNUSED",
                    "chat_model": "fake-chat",
                    "embedding_model": "fake-hash-16",
                },
                "retrieval": [
                    {"chunk_size": 200, "chunk_overlap": 20, "top_k": 1, "prompt_variant": "direct"}
                ],
                "budget": {"currency": "CNY", "hard_limit": 20},
            }
        ),
        encoding="utf-8",
    )
    return config


def run_config(config_path: Path) -> ExperimentRecord:
    config = load_experiment_config(config_path)
    dataset = load_dataset(config.dataset_path)
    chat = FakeChatProvider(
        {
            item.question: StructuredAnswer(
                answer=item.reference_answer, citations=[], abstained=False
            )
            for item in dataset.cases
        }
    )
    return run_experiment(
        config,
        ProviderBundle(embedding=FakeEmbeddingProvider(16), chat=chat),
        dataset,
    )


def test_freeze_records_hash_and_verify_accepts_untouched_holdout(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "dataset.json", base_cases())
    dataset = load_dataset(path)

    freeze = freeze_holdout(dataset, holdout_lock_path(path))
    status = verify_holdout(dataset, load_holdout_lock(holdout_lock_path(path)))

    assert holdout_lock_path(path).name == "dataset.holdout-lock.json"
    assert freeze.dataset_version == "1.0.0"
    assert freeze.holdout_case_ids == ["rag-003"]
    assert freeze.holdout_hash == dataset.holdout_hash()
    assert status.state == "frozen"
    assert status.holdout_hash == freeze.holdout_hash
    # Freezing the same content again is idempotent.
    assert freeze_holdout(dataset, holdout_lock_path(path)) == freeze
    assert len(load_holdout_lock(holdout_lock_path(path)).freezes) == 1


@pytest.mark.parametrize(
    "tampered",
    [
        [case("rag-001", "dev"), case("rag-002", "dev"), case("rag-003", "holdout", "Edited?")],
        [case("rag-001", "dev"), case("rag-002", "holdout"), case("rag-003", "holdout")],
        [case("rag-001", "dev"), case("rag-002", "dev"), case("rag-003", "dev")],
    ],
    ids=["edited-question", "case-moved-into-holdout", "holdout-case-moved-out"],
)
def test_any_change_to_a_frozen_holdout_is_refused(
    tmp_path: Path, tampered: list[dict[str, Any]]
) -> None:
    path = write_dataset(tmp_path / "dataset.json", base_cases())
    freeze_holdout(load_dataset(path), holdout_lock_path(path))
    write_dataset(path, tampered)

    with pytest.raises(HoldoutTampered, match="new dataset version"):
        verify_holdout(load_dataset(path), load_holdout_lock(holdout_lock_path(path)))
    with pytest.raises(HoldoutTampered):
        run_config(write_config(tmp_path, path))
    assert not (tmp_path / "runs.sqlite3").exists()


def test_dev_edits_keep_the_frozen_holdout_usable(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "dataset.json", base_cases())
    freeze_holdout(load_dataset(path), holdout_lock_path(path))
    edited = base_cases()
    edited[0]["question"] = "A tuned dev question?"
    write_dataset(path, edited)

    record = run_config(write_config(tmp_path, path))

    assert record.status is ExperimentStatus.COMPLETED
    assert record.identity.holdout_freeze_hash == load_dataset(path).holdout_hash()
    assert record.identity.dataset_version == "1.0.0"


def test_a_new_version_is_allowed_and_can_be_frozen_separately(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "dataset.json", base_cases())
    freeze_holdout(load_dataset(path), holdout_lock_path(path))
    changed = base_cases()
    changed[2]["question"] = "Rewritten holdout question?"
    write_dataset(path, changed, version="1.1.0")

    status = verify_holdout(load_dataset(path), load_holdout_lock(holdout_lock_path(path)))
    record = run_config(write_config(tmp_path, path))

    assert status.state == "not_frozen"
    assert record.status is ExperimentStatus.COMPLETED
    assert record.identity.holdout_freeze_hash is None
    freeze_holdout(load_dataset(path), holdout_lock_path(path))
    versions = [f.dataset_version for f in load_holdout_lock(holdout_lock_path(path)).freezes]
    assert versions == ["1.0.0", "1.1.0"]


def test_refreezing_a_version_with_different_holdout_is_refused(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "dataset.json", base_cases())
    freeze_holdout(load_dataset(path), holdout_lock_path(path))
    changed = base_cases()
    changed[2]["question"] = "Different?"
    write_dataset(path, changed)

    with pytest.raises(HoldoutTampered, match="already frozen"):
        freeze_holdout(load_dataset(path), holdout_lock_path(path))


def test_freeze_requires_holdout_cases(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "dataset.json", [case("rag-001", "dev")])

    with pytest.raises(ValueError, match="no holdout"):
        freeze_holdout(load_dataset(path), holdout_lock_path(path))


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rag_quality_lab.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_dataset_cli_shows_splits_freezes_and_verifies(tmp_path: Path) -> None:
    unlabeled = case("rag-004", None)
    path = write_dataset(tmp_path / "dataset.json", [*base_cases(), unlabeled])

    splits = run_cli("dataset", "splits", "--dataset", str(path))
    frozen = run_cli("dataset", "freeze-holdout", "--dataset", str(path))
    verified = run_cli("dataset", "verify", "--dataset", str(path))
    tampered_cases = [*base_cases(), unlabeled]
    tampered_cases[2]["reference_answer"] = "Edited answer."
    write_dataset(path, tampered_cases)
    refused = run_cli("dataset", "verify", "--dataset", str(path))

    assert splits.returncode == 0, splits.stderr
    split_payload = json.loads(splits.stdout)
    assert split_payload["dataset_version"] == "1.0.0"
    assert split_payload["splits"] == {"dev": 2, "holdout": 1, "未标注": 1}
    assert split_payload["holdout"]["state"] == "not_frozen"
    assert frozen.returncode == 0, frozen.stderr
    assert json.loads(frozen.stdout)["holdout_case_ids"] == ["rag-003"]
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["state"] == "frozen"
    assert refused.returncode == 1
    refused_payload = json.loads(refused.stdout)
    assert refused_payload["state"] == "tampered"
    assert "new dataset version" in refused_payload["error"]


def test_split_summary_counts_labels_and_reports_tampering(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "dataset.json", [*base_cases(), case("rag-004", None)])
    freeze_holdout(load_dataset(path), holdout_lock_path(path))
    lock = load_holdout_lock(holdout_lock_path(path))

    summary = split_summary(load_dataset(path), lock)
    tampered = base_cases()
    tampered[2]["question"] = "Edited?"
    write_dataset(path, tampered)
    refused = split_summary(load_dataset(path), lock)

    assert summary["splits"] == {"dev": 2, "holdout": 1, "未标注": 1}
    assert summary["review_status"] == {"未标注": 4}
    assert summary["case_ids_by_split"] == {"dev": ["rag-001", "rag-002"], "holdout": ["rag-003"]}
    assert summary["holdout"]["state"] == "frozen"  # type: ignore[index]
    assert refused["holdout"]["state"] == "tampered"  # type: ignore[index]
