"""Interrupted live experiments resume without losing, resetting, or repeating spend.

Each kill test runs the experiment in a child process through the real
``OpenAICompatibleProvider`` and an in-process HTTP double that durably logs
every request, then kills the child without any cleanup. Request counts in the
log are the evidence that completed work is never sent twice.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_quality_lab.config import load_dataset, load_experiment_config
from rag_quality_lab.domain.models import (
    Answerability,
    ExperimentConfig,
    ExperimentRecord,
    ExperimentStatus,
    PricingConfig,
    StructuredAnswer,
)
from rag_quality_lab.experiments.budget import planned_call_cost
from rag_quality_lab.experiments.runner import (
    ProviderBundle,
    ResumeRefused,
    planned_calls,
    resume_experiment,
    run_experiment,
)
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.providers.fake import FakeChatProvider, FakeEmbeddingProvider
from rag_quality_lab.reporting import generate_reports
from rag_quality_lab.retrieval.index import load_documents
from tests.experiments.resume_harness import (
    API_KEY_ENV,
    CASES,
    CORPUS_TEXT,
    LoggedSession,
    generation_only_prices,
    kill_when_frozen,
    read_log,
    remote_bundle,
    start_child,
    write_inputs,
)

# Per-case generation output cap with max_retries=0; pricing it at this rate
# makes every case cost exactly 0.2 CNY, its full reservation.
CASE_COST_RATE = 195.3125
TIMING_FIELDS = {"latency_ms", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"}


@pytest.fixture(autouse=True)
def test_only_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-only-not-a-real-key")


def only_experiment_id(database: Path) -> str:
    with ExperimentStore(database) as store:
        rows = store.connection.execute("SELECT id FROM experiments").fetchall()
    assert len(rows) == 1
    return str(rows[0][0])


def interrupt(
    workdir: Path,
    *freeze: str,
    generation_completion_tokens: int = 20,
    **inputs: Any,
) -> tuple[ExperimentConfig, str, Path]:
    """Run the harness in a child process, kill it when frozen, return its state."""

    config_path = write_inputs(workdir, **inputs)
    log = workdir / "requests.jsonl"
    marker = workdir / "frozen.marker"
    child = start_child(
        config_path,
        log,
        marker,
        *freeze,
        generation_completion_tokens=generation_completion_tokens,
    )
    kill_when_frozen(child, marker)
    config = load_experiment_config(config_path)
    return config, only_experiment_id(config.database_path), log


def resume(
    config: ExperimentConfig,
    experiment_id: str,
    log: Path,
    *,
    retry_unknown: bool = False,
    generation_completion_tokens: int = 20,
) -> ExperimentRecord:
    session = LoggedSession(log, generation_completion_tokens=generation_completion_tokens)
    return resume_experiment(
        experiment_id,
        config,
        remote_bundle(session, max_retries=config.provider.max_retries),
        load_dataset(config.dataset_path),
        retry_unknown=retry_unknown,
    )


def spent(config: ExperimentConfig, experiment_id: str) -> Decimal:
    with ExperimentStore(config.database_path) as store:
        store.reap_orphans()
        return store.ledger_spent(experiment_id)


def status(config: ExperimentConfig, experiment_id: str) -> ExperimentStatus:
    with ExperimentStore(config.database_path) as store:
        store.reap_orphans()
        return store.get_experiment(experiment_id).status


def per_case_reservation(config: ExperimentConfig, phase: str) -> Decimal:
    assert config.pricing_path is not None
    pricing = PricingConfig.model_validate(
        yaml.safe_load(config.pricing_path.read_text(encoding="utf-8"))
    )
    call = next(
        call
        for call in planned_calls(
            config,
            load_dataset(config.dataset_path),
            load_documents(config.knowledge_base_path),
        )
        if call.phase == phase
    )
    return planned_call_cost(call, pricing) / call.count


def comparable_cases(record: ExperimentRecord, artifact_dir: Path) -> list[dict[str, Any]]:
    report = json.loads(generate_reports(record, artifact_dir).json.read_text("utf-8"))
    cases = [
        {key: value for key, value in case.items() if key not in TIMING_FIELDS}
        for case in report["case_results"]
    ]
    return sorted(cases, key=lambda case: (case["config_id"], case["case_id"]))


def test_killed_run_resumes_without_repeating_completed_cases(tmp_path: Path) -> None:
    config, experiment_id, log = interrupt(
        tmp_path / "interrupted", "--block-after-results", "2"
    )
    first_requests = read_log(log)
    first_spent = spent(config, experiment_id)
    assert status(config, experiment_id) is ExperimentStatus.INTERRUPTED
    finished_before_kill = {"rag-001", "rag-002"}
    assert {entry["case_id"] for entry in first_requests} - {None} == finished_before_kill

    resumed = resume(config, experiment_id, log)

    resumed_requests = read_log(log)[len(first_requests) :]
    assert resumed.status is ExperimentStatus.COMPLETED
    assert [entry for entry in resumed_requests if entry["case_id"] in finished_before_kill] == []
    assert [entry for entry in resumed_requests if entry["kind"] == "embedding_index"] == []
    assert {entry["case_id"] for entry in resumed_requests} == {"rag-003", "rag-004", "rag-005"}
    total_spent = spent(config, experiment_id)
    resumed_spent = total_spent - first_spent
    assert first_spent > 0 and resumed_spent > 0
    assert total_spent == first_spent + resumed_spent
    assert Decimal(str(resumed.summary["total_cost"])) == total_spent

    one_shot_config = load_experiment_config(write_inputs(tmp_path / "one-shot"))
    one_shot_log = tmp_path / "one-shot" / "requests.jsonl"
    one_shot = run_experiment(
        one_shot_config,
        remote_bundle(LoggedSession(one_shot_log)),
        load_dataset(one_shot_config.dataset_path),
    )
    assert len(read_log(one_shot_log)) == len(read_log(log))
    assert spent(one_shot_config, one_shot.id) == total_spent
    assert comparable_cases(resumed, tmp_path / "resumed-report") == comparable_cases(
        one_shot, tmp_path / "one-shot-report"
    )
    assert {
        key: value for key, value in resumed.summary.items() if key not in TIMING_FIELDS
    } == {key: value for key, value in one_shot.summary.items() if key not in TIMING_FIELDS}


def test_call_killed_before_settlement_is_charged_at_cap_and_not_resent(
    tmp_path: Path,
) -> None:
    config, experiment_id, log = interrupt(
        tmp_path, "--block-generation-case", "rag-003"
    )
    first_requests = read_log(log)
    assert first_requests[-1] == {
        "path": "/chat/completions",
        "kind": "generation",
        "case_id": "rag-003",
    }

    with ExperimentStore(config.database_path) as store:
        assert store.reap_orphans() == [experiment_id]
        entries = [
            entry for entry in store.ledger_entries(experiment_id)
            if entry.case_id == "rag-003"
        ]
    by_phase = {entry.phase: entry for entry in entries}
    unknown_charge = per_case_reservation(
        config, "generation_with_repair"
    ) + per_case_reservation(config, "embedding_query")
    assert by_phase["generation_with_repair"].state == "unknown"
    assert by_phase["embedding_query"].state == "unknown"
    assert {by_phase[phase].state for phase in ("judge_with_repair", "embedding_answer")} == {
        "released"
    }
    assert sum((entry.charged for entry in entries), Decimal("0")) == unknown_charge
    first_spent = spent(config, experiment_id)

    status_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_quality_lab.cli",
            "status",
            "--experiment",
            experiment_id,
            "--database",
            str(config.database_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert status_result.returncode == 0, status_result.stderr
    payload = json.loads(status_result.stdout)
    assert payload["status"] == "interrupted"
    assert payload["progress"] == {"done": 2, "total": 5, "unknown": 1}
    assert Decimal(payload["spent"]) == first_spent

    resumed = resume(config, experiment_id, log)

    resumed_requests = read_log(log)[len(first_requests) :]
    assert resumed.status is ExperimentStatus.INCOMPLETE
    assert "rag-003" not in {entry["case_id"] for entry in resumed_requests}
    assert {entry["case_id"] for entry in resumed_requests} == {"rag-004", "rag-005"}
    assert "rag-003" not in {case.case_id for case in resumed.case_results}
    assert {call.case_id for call in resumed.unknown_calls} == {"rag-003"}
    assert resumed.summary["unknown_outcome_count"] == 1.0
    assert Decimal(str(resumed.summary["unknown_outcome_cost"])) == unknown_charge
    assert Decimal(str(resumed.summary["total_cost"])) == spent(config, experiment_id)

    request_count = len(read_log(log))
    with pytest.raises(ResumeRefused, match="--retry-unknown"):
        resume(config, experiment_id, log)
    assert len(read_log(log)) == request_count
    assert status(config, experiment_id) is ExperimentStatus.INCOMPLETE

    retried = resume(config, experiment_id, log, retry_unknown=True)

    assert retried.status is ExperimentStatus.COMPLETED
    assert {entry["case_id"] for entry in read_log(log)[request_count:]} == {"rag-003"}
    assert "rag-003" in {case.case_id for case in retried.case_results}
    assert Decimal(str(retried.summary["total_cost"])) == spent(config, experiment_id)


def test_retry_unknown_resends_the_call_and_charges_it_separately(
    tmp_path: Path,
) -> None:
    config, experiment_id, log = interrupt(
        tmp_path, "--block-generation-case", "rag-003"
    )
    first_requests = read_log(log)
    first_spent = spent(config, experiment_id)

    resumed = resume(config, experiment_id, log, retry_unknown=True)

    resumed_requests = read_log(log)[len(first_requests) :]
    generation_attempts = [
        entry
        for entry in read_log(log)
        if entry["kind"] == "generation" and entry["case_id"] == "rag-003"
    ]
    assert resumed.status is ExperimentStatus.COMPLETED
    assert len(generation_attempts) == 2
    assert {entry["case_id"] for entry in resumed_requests} == {
        "rag-003",
        "rag-004",
        "rag-005",
    }
    retried = next(case for case in resumed.case_results if case.case_id == "rag-003")
    assert retried.status == "completed"
    retried_embeddings = sum(
        (
            call.cost
            for call in resumed.embedding_calls
            if call.case_id in {"rag-003", "rag-004", "rag-005"}
        ),
        Decimal("0"),
    )
    resumed_case_cost = sum(
        (
            case.cost
            for case in resumed.case_results
            if case.case_id in {"rag-003", "rag-004", "rag-005"}
        ),
        Decimal("0"),
    )
    total_spent = spent(config, experiment_id)
    assert total_spent == first_spent + resumed_case_cost + retried_embeddings
    assert {call.case_id for call in resumed.unknown_calls} == {"rag-003"}
    assert Decimal(str(resumed.summary["total_cost"])) == total_spent


def test_resume_refuses_changed_dataset_corpus_config_or_pricing(tmp_path: Path) -> None:
    config, experiment_id, log = interrupt(tmp_path, "--block-after-results", "1")
    request_count = len(read_log(log))
    dataset_path = config.dataset_path
    original_dataset = dataset_path.read_text(encoding="utf-8")
    changed = json.loads(original_dataset)
    changed["cases"][4]["reference_answer"] = "Chunkers."
    dataset_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ResumeRefused, match="dataset"):
        resume(config, experiment_id, log)

    dataset_path.write_text(original_dataset, encoding="utf-8")
    corpus = config.knowledge_base_path / "doc-01-rag.md"
    corpus.write_text(CORPUS_TEXT + " Extra.", encoding="utf-8")
    with pytest.raises(ResumeRefused, match="corpus"):
        resume(config, experiment_id, log)

    corpus.write_text(CORPUS_TEXT, encoding="utf-8")
    changed_config = config.model_copy(
        update={
            "retrieval": [
                config.retrieval[0].model_copy(update={"top_k": 2}),
            ]
        }
    )
    with pytest.raises(ResumeRefused, match=r"config\.retrieval\[0\]\.top_k"):
        resume(changed_config, experiment_id, log)

    assert config.pricing_path is not None
    original_pricing = config.pricing_path.read_text(encoding="utf-8")
    pricing = yaml.safe_load(original_pricing)
    pricing["models"]["remote-chat"]["output"] = 3
    config.pricing_path.write_text(yaml.safe_dump(pricing), encoding="utf-8")
    with pytest.raises(ResumeRefused, match="pricing"):
        resume(config, experiment_id, log)

    assert len(read_log(log)) == request_count
    assert status(config, experiment_id) is ExperimentStatus.INTERRUPTED
    config.pricing_path.write_text(original_pricing, encoding="utf-8")
    assert resume(config, experiment_id, log).status is ExperimentStatus.COMPLETED


def test_restored_budget_allows_only_the_unspent_remainder(tmp_path: Path) -> None:
    config, experiment_id, log = interrupt(
        tmp_path,
        "--block-generation-case",
        "rag-004",
        generation_completion_tokens=1024,
        prices=generation_only_prices(CASE_COST_RATE),
        hard_limit="1.0",
        preflight_fraction="1",
        safety_multiplier="1",
        max_retries=0,
    )
    assert spent(config, experiment_id) == Decimal("0.8")
    request_count = len(read_log(log))

    with pytest.raises(ResumeRefused, match="budget"):
        resume(
            config,
            experiment_id,
            log,
            retry_unknown=True,
            generation_completion_tokens=1024,
        )
    assert len(read_log(log)) == request_count
    assert status(config, experiment_id) is ExperimentStatus.INTERRUPTED

    resumed = resume(config, experiment_id, log, generation_completion_tokens=1024)

    assert resumed.status is ExperimentStatus.INCOMPLETE
    assert {entry["case_id"] for entry in read_log(log)[request_count:]} == {"rag-005"}
    assert spent(config, experiment_id) == Decimal("1.0")
    assert Decimal(str(resumed.summary["total_cost"])) == Decimal("1.0")
    with pytest.raises(ResumeRefused, match="budget"):
        resume(
            config,
            experiment_id,
            log,
            retry_unknown=True,
            generation_completion_tokens=1024,
        )


def run_and_cancel_while_blocked(
    tmp_path: Path, blocked_kind: str
) -> tuple[ExperimentConfig, ExperimentRecord, Path, int, subprocess.CompletedProcess[str]]:
    """Run in a thread, cancel through the CLI while rag-001 blocks in one request."""

    config = load_experiment_config(write_inputs(tmp_path))
    log = tmp_path / "requests.jsonl"
    blocked = threading.Event()
    release = threading.Event()

    def hold(entry: dict[str, Any]) -> None:
        if entry["kind"] == blocked_kind and entry["case_id"] == "rag-001":
            blocked.set()
            assert release.wait(60)

    session = LoggedSession(log, on_request=hold)
    outcome: list[ExperimentRecord] = []
    worker = threading.Thread(
        target=lambda: outcome.append(
            run_experiment(config, remote_bundle(session), load_dataset(config.dataset_path))
        )
    )
    worker.start()
    assert blocked.wait(60)
    experiment_id = only_experiment_id(config.database_path)
    cancel = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_quality_lab.cli",
            "cancel",
            "--experiment",
            experiment_id,
            "--database",
            str(config.database_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    requests_at_cancel = len(read_log(log))
    release.set()
    worker.join(60)
    assert not worker.is_alive()
    return config, outcome[0], log, requests_at_cancel, cancel


def assert_cancelled_case_is_settled_but_unscored(
    config: ExperimentConfig, record: ExperimentRecord
) -> None:
    assert record.status is ExperimentStatus.CANCELLED
    assert [case.case_id for case in record.case_results] == ["rag-001"]
    cancelled = record.case_results[0]
    assert cancelled.status == "cancelled"
    assert cancelled.metrics == {}
    assert cancelled.judge is None
    assert record.summary["completed_cases"] == 0.0
    assert record.summary["failure_count"] == 0.0
    assert record.summary["cancelled_case_count"] == 1.0
    with ExperimentStore(config.database_path) as store:
        entries = {entry.phase: entry for entry in store.ledger_entries(record.id)}
    assert entries["generation_with_repair"].state == "settled"
    assert entries["generation_with_repair"].charged == cancelled.cost > 0
    assert entries["judge_with_repair"].state == "released"
    assert entries["judge_with_repair"].charged == 0
    assert Decimal(str(record.summary["total_cost"])) == spent(config, record.id)
    report = json.loads(
        generate_reports(record, config.artifact_dir).json.read_text("utf-8")
    )
    assert report["failures"] == []
    assert report["system"]["failure_count"] == 0
    assert [case["case_id"] for case in report["cancelled_cases"]] == ["rag-001"]


def test_cancel_during_generation_sends_no_further_phase(tmp_path: Path) -> None:
    config, record, log, requests_at_cancel, cancel = run_and_cancel_while_blocked(
        tmp_path, "generation"
    )

    assert cancel.returncode == 0, cancel.stderr
    assert json.loads(cancel.stdout)["cancel_requested"] is True
    assert read_log(log)[requests_at_cancel:] == []
    assert_cancelled_case_is_settled_but_unscored(config, record)
    with ExperimentStore(config.database_path) as store:
        entries = {entry.phase: entry for entry in store.ledger_entries(record.id)}
    assert entries["embedding_answer"].state == "released"

    with pytest.raises(ResumeRefused, match="cancelled"):
        resume(config, record.id, log)


def test_cancel_after_generation_returns_never_sends_the_judge(tmp_path: Path) -> None:
    config, record, log, requests_at_cancel, cancel = run_and_cancel_while_blocked(
        tmp_path, "embedding_answer"
    )

    assert cancel.returncode == 0, cancel.stderr
    requests = read_log(log)
    assert requests[requests_at_cancel:] == []
    assert [entry for entry in requests if entry["kind"] == "judge"] == []
    assert [entry["kind"] for entry in requests if entry["case_id"] == "rag-001"] == [
        "embedding_query",
        "generation",
        "embedding_answer",
    ]
    assert_cancelled_case_is_settled_but_unscored(config, record)
    with ExperimentStore(config.database_path) as store:
        entries = {entry.phase: entry for entry in store.ledger_entries(record.id)}
    assert entries["embedding_answer"].state == "settled"
    assert entries["embedding_answer"].charged > 0


def test_resume_of_completed_experiment_is_refused(tmp_path: Path) -> None:
    config = load_experiment_config(write_inputs(tmp_path))
    log = tmp_path / "requests.jsonl"
    record = run_experiment(
        config, remote_bundle(LoggedSession(log)), load_dataset(config.dataset_path)
    )
    request_count = len(read_log(log))

    with pytest.raises(ResumeRefused, match="completed"):
        resume(config, record.id, log)

    assert len(read_log(log)) == request_count


class InterruptingChatProvider(FakeChatProvider):
    """Mock chat provider that simulates Ctrl+C on one question."""

    def __init__(self, answers: dict[str, StructuredAnswer], interrupt_on: str) -> None:
        super().__init__(answers)
        self.interrupt_on = interrupt_on

    def answer(self, question: str, *args: Any, **kwargs: Any) -> Any:
        if question == self.interrupt_on:
            raise KeyboardInterrupt
        return super().answer(question, *args, **kwargs)


def test_mock_run_interrupted_by_keyboard_resumes_to_the_one_shot_result(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(write_inputs(tmp_path)).model_copy(
        update={"mode": "mock", "pricing_path": None}
    )
    dataset = load_dataset(config.dataset_path)
    answers = {
        case.question: StructuredAnswer(
            answer=case.reference_answer,
            citations=case.expected_document_ids,
            abstained=case.answerability is not Answerability.ANSWERABLE,
        )
        for case in dataset.cases
    }
    config = config.model_copy(
        update={
            "provider": config.provider.model_copy(
                update={"embedding_model": "fake-hash-32", "judge_model": None}
            )
        }
    )
    interrupting = ProviderBundle(
        embedding=FakeEmbeddingProvider(32),
        chat=InterruptingChatProvider(answers, CASES[2]["question"]),
    )

    with pytest.raises(KeyboardInterrupt):
        run_experiment(config, interrupting, dataset)

    experiment_id = only_experiment_id(config.database_path)
    assert status(config, experiment_id) is ExperimentStatus.INTERRUPTED
    bundle = ProviderBundle(embedding=FakeEmbeddingProvider(32), chat=FakeChatProvider(answers))
    resumed = resume_experiment(experiment_id, config, bundle, dataset)
    one_shot = run_experiment(
        config.model_copy(update={"database_path": tmp_path / "one-shot.sqlite3"}),
        bundle,
        dataset,
    )

    assert resumed.status is ExperimentStatus.COMPLETED
    assert comparable_cases(resumed, tmp_path / "a") == comparable_cases(
        one_shot, tmp_path / "b"
    )
