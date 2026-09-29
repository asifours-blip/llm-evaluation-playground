"""Thin read/write facade the workbench API calls into.

Every function here is a direct call into the existing service layer
(config loaders, :mod:`rag_quality_lab.experiments`, and
:mod:`rag_quality_lab.reporting`); nothing here re-implements planning,
budgeting, running, or report rendering. The web layer's only job is to
shape those calls' results into JSON-friendly dictionaries and to run a
live/mock experiment on a background thread so an HTTP request can return
before the run finishes.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rag_quality_lab.cli import _live_preflight, _provider_bundle
from rag_quality_lab.config import load_dataset, load_experiment_config
from rag_quality_lab.config.holdout import (
    holdout_lock_path,
    load_holdout_lock,
    split_summary,
)
from rag_quality_lab.domain.models import ExperimentConfig
from rag_quality_lab.experiments.compare import compare_arms, compare_experiments
from rag_quality_lab.experiments.runner import planned_calls, resume_experiment
from rag_quality_lab.experiments.runner import run_experiment as _run_experiment
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.reporting import generate_reports
from rag_quality_lab.reporting.report import ReportPaths, generate_comparison_report
from rag_quality_lab.retrieval.index import load_documents
from rag_quality_lab.task_eval.core import compare as compare_task_results
from rag_quality_lab.task_eval.core import load_result


class LiveRunNotAllowed(PermissionError):
    """Raised when a live run is attempted while the server disallows it."""


class PathEscapesArtifactDir(ValueError):
    """Raised when a requested artifact path would resolve outside its root."""


def dataset_view(dataset_path: Path) -> dict[str, Any]:
    """Splits, labels, and holdout state for one dataset file."""

    dataset = load_dataset(dataset_path)
    return split_summary(dataset, load_holdout_lock(holdout_lock_path(dataset_path)))


def config_view(config_path: Path) -> dict[str, Any]:
    """A summary of one experiment configuration: mode, arms, and paths."""

    config = load_experiment_config(config_path)
    return {
        "name": config.name,
        "mode": config.mode,
        "dataset_path": str(config.dataset_path),
        "database_path": str(config.database_path),
        "artifact_dir": str(config.artifact_dir),
        "splits": list(config.splits) if config.splits is not None else None,
        "random_seed": config.random_seed,
        "budget": {
            "currency": config.budget.currency,
            "hard_limit": str(config.budget.hard_limit),
        },
        "arms": [arm.config_id for arm in config.retrieval],
    }


def precheck(config_path: Path) -> dict[str, Any]:
    """Planned request counts, and estimated cost for live configurations.

    Mock configurations send no paid request, so only the request plan is
    returned; live configurations reuse the same preflight the CLI's
    ``run --preflight-only`` performs, including the budget decision.
    """

    config = load_experiment_config(config_path)
    dataset = load_dataset(config.dataset_path)
    if config.mode == "live":
        decision = _live_preflight(config, dataset)
        return decision.model_dump(mode="json")
    documents = load_documents(config.knowledge_base_path)
    plan = planned_calls(config, dataset.select_splits(config.splits), documents)
    return {
        "mode": "mock",
        "allowed": True,
        "planned_calls": [call.model_dump(mode="json") for call in plan],
        "total_request_count": sum(call.count * call.requests_per_case for call in plan),
    }


def list_experiments(database_path: Path) -> list[dict[str, Any]]:
    with ExperimentStore(database_path) as store:
        store.reap_orphans()
        return [store.progress(experiment_id) for experiment_id in store.experiment_ids()]


def experiment_detail(database_path: Path, experiment_id: str) -> dict[str, Any]:
    with ExperimentStore(database_path) as store:
        store.reap_orphans()
        resolved_id = store.resolve_experiment_id(experiment_id)
        progress = store.progress(resolved_id)
        record = store.get_experiment(resolved_id)
    failures = [
        {
            "case_id": result.case_id,
            "config_id": result.config_id,
            "model": result.model,
            "status": result.status,
            "error": result.error,
        }
        for result in record.case_results
        if result.status not in {"completed", "cancelled"}
    ]
    return {**progress, "failures": failures, "case_result_count": len(record.case_results)}


def case_results(
    database_path: Path,
    experiment_id: str,
    *,
    config_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Page through one experiment's per-question results and evidence."""

    with ExperimentStore(database_path) as store:
        resolved_id = store.resolve_experiment_id(experiment_id)
        record = store.get_experiment(resolved_id)
    results = record.case_results
    if config_id is not None:
        results = [result for result in results if result.config_id == config_id]
    if status is not None:
        results = [result for result in results if result.status == status]
    page = results[offset : offset + limit]
    return {
        "total": len(results),
        "offset": offset,
        "limit": limit,
        "results": [
            {
                "case_id": result.case_id,
                "config_id": result.config_id,
                "model": result.model,
                "status": result.status,
                "split": result.split,
                "difficulty": result.difficulty,
                "category": result.category,
                "question": result.question,
                "reference_answer": result.reference_answer,
                "answer": result.answer.model_dump(mode="json") if result.answer else None,
                "judge": result.judge.model_dump(mode="json") if result.judge else None,
                "metrics": result.metrics,
                "evidence": [
                    {
                        "document_id": hit.chunk.document_id,
                        "chunk_id": hit.chunk.id,
                        "score": hit.score,
                        "text": hit.chunk.text,
                    }
                    for hit in result.retrieval_hits
                ],
                "error": result.error,
            }
            for result in page
        ],
    }


def request_cancel(database_path: Path, experiment_id: str) -> dict[str, Any]:
    with ExperimentStore(database_path) as store:
        resolved_id = store.resolve_experiment_id(experiment_id)
        status = store.request_cancel(resolved_id)
    return {"experiment_id": resolved_id, "status": status.value}


def generate_report(
    database_path: Path,
    experiment_id: str,
    output_dir: Path,
    *,
    badge: Literal["mock", "pilot", "final"] | None = None,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Regenerate a stored report; identical to ``rag-quality report``.

    Calls the same :func:`generate_reports` the CLI calls, with the same
    inputs, so the JSON and HTML bytes it writes are byte-identical to what
    the CLI would write for the same experiment, output directory, badge,
    and baseline.
    """

    with ExperimentStore(database_path) as store:
        resolved_id = store.resolve_experiment_id(experiment_id)
        record = store.get_experiment(resolved_id)
        comparison = None
        if baseline is not None:
            baseline_record = store.get_experiment(store.resolve_experiment_id(baseline))
            comparison = compare_experiments(baseline_record, record)
        paths = generate_reports(record, output_dir, badge=badge, comparison=comparison)
        store.record_artifact(
            resolved_id, kind="json_report", path=str(paths.json), sha256=paths.json_sha256
        )
        store.record_artifact(
            resolved_id, kind="html_report", path=str(paths.html), sha256=paths.html_sha256
        )
    return _report_paths_payload(resolved_id, paths)


def generate_pair_report(
    database_path: Path,
    baseline: str,
    candidate: str,
    baseline_config: str,
    candidate_config: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Regenerate a paired comparison report; identical to ``rag-quality compare``."""

    with ExperimentStore(database_path) as store:
        baseline_record = store.get_experiment(store.resolve_experiment_id(baseline))
        candidate_record = store.get_experiment(store.resolve_experiment_id(candidate))
    paired = compare_arms(
        baseline_record, candidate_record, baseline_config, candidate_config
    )
    paths = generate_comparison_report(paired, output_dir)
    return {
        "report_json": str(paths.json.resolve()),
        "report_html": str(paths.html.resolve()),
        "json_sha256": paths.json_sha256,
        "html_sha256": paths.html_sha256,
        "warning": paired.warning,
    }


def task_result_view(report_path: Path) -> dict[str, Any]:
    """Read an already-scored task artifact; no evaluation or writing on GET."""
    return load_result(report_path).model_dump(mode="json")


def task_compare_view(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    return compare_task_results(load_result(baseline_path), load_result(candidate_path))


def artifact_download_path(artifact_dir: Path, requested: str) -> Path:
    """Resolve ``requested`` under ``artifact_dir``, refusing any path escape."""

    base = artifact_dir.resolve()
    candidate = (base / requested).resolve()
    if candidate != base and base not in candidate.parents:
        raise PathEscapesArtifactDir(
            f"requested path escapes the artifact directory: {requested}"
        )
    if not candidate.is_file():
        raise FileNotFoundError(requested)
    return candidate


def _report_paths_payload(experiment_id: str, paths: ReportPaths) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "report_json": str(paths.json.resolve()),
        "report_html": str(paths.html.resolve()),
        "json_sha256": paths.json_sha256,
        "html_sha256": paths.html_sha256,
    }


@dataclass
class RunState:
    """In-memory status of one background run started by this server."""

    status: Literal["starting", "running", "completed", "failed"] = "starting"
    experiment_id: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None


class RunManager:
    """Runs experiments on background threads and tracks their status.

    The store is the single source of truth for progress; this registry
    only remembers which background thread belongs to which client-visible
    token, and how it ended, since a fresh ``run_experiment`` call does not
    expose its experiment id until the whole run finishes.
    """

    def __init__(self, *, allow_live: bool = False) -> None:
        self._allow_live = allow_live
        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {}

    def allow_live(self) -> bool:
        return self._allow_live

    def get(self, token: str) -> RunState:
        with self._lock:
            state = self._runs.get(token)
        if state is None:
            raise KeyError(token)
        return state

    def start_run(self, config_path: Path, *, confirm_live_run: bool) -> str:
        config = load_experiment_config(config_path)
        self._check_live_allowed(config, confirm_live_run)
        token = str(uuid.uuid4())
        state = RunState()
        with self._lock:
            self._runs[token] = state
        with ExperimentStore(config.database_path) as store:
            existing_ids = set(store.experiment_ids())

        def target() -> None:
            dataset = load_dataset(config.dataset_path)
            try:
                record = _run_experiment(config, _provider_bundle(config, dataset), dataset)
            except Exception as error:  # noqa: BLE001 - surfaced to the client, not swallowed
                state.status = "failed"
                state.error = str(error)
                return
            state.experiment_id = record.id
            state.status = "completed"
            state.summary = record.summary

        # Mark running before the worker starts: a fast run can finish while
        # the new experiment ID is being awaited, and its final status must
        # not be overwritten afterwards.
        state.status = "running"
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        discovered_id = _await_new_experiment_id(
            config.database_path, existing_ids, timeout_seconds=5.0
        )
        if state.experiment_id is None:
            state.experiment_id = discovered_id
        return token

    def resume_run(
        self,
        config_path: Path,
        experiment_id: str,
        *,
        confirm_live_run: bool,
        retry_unknown: bool = False,
    ) -> str:
        config = load_experiment_config(config_path)
        self._check_live_allowed(config, confirm_live_run)
        with ExperimentStore(config.database_path) as store:
            resolved_id = store.resolve_experiment_id(experiment_id)
        token = str(uuid.uuid4())
        state = RunState(experiment_id=resolved_id, status="running")
        with self._lock:
            self._runs[token] = state

        def target() -> None:
            dataset = load_dataset(config.dataset_path)
            try:
                record = resume_experiment(
                    resolved_id,
                    config,
                    _provider_bundle(config, dataset),
                    dataset,
                    retry_unknown=retry_unknown,
                )
            except Exception as error:  # noqa: BLE001
                state.status = "failed"
                state.error = str(error)
                return
            state.status = "completed"
            state.summary = record.summary

        threading.Thread(target=target, daemon=True).start()
        return token

    def _check_live_allowed(self, config: ExperimentConfig, confirm_live_run: bool) -> None:
        if config.mode != "live":
            return
        if not confirm_live_run:
            raise ValueError("live runs require confirm_live_run")
        if not self._allow_live:
            raise LiveRunNotAllowed(
                "this server was started without --allow-live; live runs are refused"
            )


def _await_new_experiment_id(
    database_path: Path, existing_ids: set[str], *, timeout_seconds: float
) -> str | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with ExperimentStore(database_path) as store:
            for candidate in store.experiment_ids():
                if candidate not in existing_ids:
                    return candidate
        time.sleep(0.02)
    return None
