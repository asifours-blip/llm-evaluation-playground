from decimal import Decimal
from pathlib import Path

import pytest

from rag_quality_lab.domain.models import (
    CaseResult,
    Chunk,
    ExperimentIdentity,
    ExperimentStatus,
    RetrievalHit,
    StructuredAnswer,
    TokenUsage,
)
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.metrics.calibration import HumanAnnotation


def example_identity() -> ExperimentIdentity:
    return ExperimentIdentity(
        name="offline-baseline",
        mode="mock",
        commit_sha="abc123",
        dirty=False,
        dataset_hash="dataset-hash",
        prompt_hashes={"direct": "prompt-hash"},
        config={"seed": 42},
        random_seed=42,
        python_version="3.11.9",
    )


def example_case_result() -> CaseResult:
    return CaseResult(
        case_id="rag-001",
        config_id="chunk400-top3-direct",
        model="fake-model",
        answer=StructuredAnswer(
            answer="RAG retrieves evidence.",
            citations=["doc-01"],
            abstained=False,
        ),
        retrieval_hits=[
            RetrievalHit(
                chunk=Chunk(
                    id="doc-01#chunk-000",
                    document_id="doc-01",
                    text="RAG retrieves evidence.",
                ),
                score=0.9,
            )
        ],
        metrics={"recall_at_k": 1.0, "answer_f1": 1.0},
        usage=TokenUsage(input_tokens=10, output_tokens=4),
        latency_ms=12.5,
        cost=Decimal("0.0001"),
        status="completed",
    )


def test_store_enables_wal_busy_timeout_and_schema(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        assert str(store.pragma("journal_mode")).lower() == "wal"
        assert int(store.pragma("busy_timeout")) == 5000
        assert int(store.pragma("foreign_keys")) == 1
        assert {
            "experiments",
            "case_runs",
            "retrieval_hits",
            "metric_results",
            "artifacts",
            "human_annotations",
        } <= store.table_names()


def test_completed_experiment_round_trips_typed_results(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        experiment_id = store.create_experiment(example_identity())
        store.record_case_result(experiment_id, example_case_result())
        store.finish_experiment(
            experiment_id,
            ExperimentStatus.COMPLETED,
            summary={"answer_f1": 1.0},
        )

        loaded = store.get_experiment(experiment_id)

    assert loaded.status is ExperimentStatus.COMPLETED
    assert loaded.identity.dataset_hash == "dataset-hash"
    assert len(loaded.case_results) == 1
    assert loaded.case_results[0].answer is not None
    assert loaded.case_results[0].answer.citations == ["doc-01"]
    assert loaded.case_results[0].cost == Decimal("0.0001")
    assert loaded.summary == {"answer_f1": 1.0}


def test_completed_case_keys_support_resume_and_reject_duplicates(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        experiment_id = store.create_experiment(example_identity())
        result = example_case_result()
        store.record_case_result(experiment_id, result)

        assert store.completed_case_keys(experiment_id) == {
            (result.case_id, result.config_id, result.model)
        }
        with pytest.raises(ValueError, match="duplicate"):
            store.record_case_result(experiment_id, result)


def test_terminal_experiment_rejects_another_transition(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        experiment_id = store.create_experiment(example_identity())
        store.finish_experiment(experiment_id, ExperimentStatus.FAILED)

        with pytest.raises(ValueError, match="transition"):
            store.finish_experiment(experiment_id, ExperimentStatus.COMPLETED)


def test_live_alias_artifacts_and_human_annotations_are_persisted(
    tmp_path: Path,
) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        store.create_experiment(example_identity())
        live_id = store.create_experiment(
            example_identity().model_copy(update={"name": "live-pilot", "mode": "live"})
        )

        assert store.resolve_experiment_id("latest-live") == live_id

        store.record_artifact(
            live_id,
            kind="html_report",
            path="report.html",
            sha256="abc123",
            metadata={"badge": "pilot"},
        )
        store.record_human_annotations(
            live_id,
            [HumanAnnotation(case_id="rag-001", human_score=4)],
        )

        assert store.get_human_annotations(live_id) == [
            HumanAnnotation(case_id="rag-001", human_score=4)
        ]
        artifact = store.connection.execute(
            "SELECT kind, path, sha256, metadata_json FROM artifacts"
        ).fetchone()

    assert artifact is not None
    assert tuple(artifact) == (
        "html_report",
        "report.html",
        "abc123",
        '{"badge":"pilot"}',
    )
