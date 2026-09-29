"""Versioned inputs and deterministic complete-task scoring.

The suite owns expectations. Observations are untrusted records of what happened.
No model call or upstream agent execution occurs here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_quality_lab.domain.models import canonical_hash


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    role: Literal["authoritative", "conflicting"]


class Expected(StrictModel):
    artifact: dict[str, Any]
    citation_ids: list[str]
    final_state: dict[str, Any]


class Task(StrictModel):
    task_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    evidence: list[Evidence]
    expected: Expected

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate evidence_id in {self.task_id}")
        authoritative = {item.evidence_id for item in self.evidence if item.role == "authoritative"}
        if not set(self.expected.citation_ids) <= authoritative:
            raise ValueError(f"expected citations must be authoritative in {self.task_id}")
        if len(self.expected.citation_ids) != len(set(self.expected.citation_ids)):
            raise ValueError(f"duplicate expected citation in {self.task_id}")
        return self


class Suite(StrictModel):
    schema_version: Literal["task-eval/v1"]
    suite_id: str = Field(min_length=1)
    suite_version: str = Field(min_length=1)
    source_kind: Literal["synthetic_demo", "reviewed_external"]
    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("suite task_id values must be unique")
        return self

    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class Observation(StrictModel):
    task_id: str = Field(min_length=1)
    retrieved_evidence_ids: list[str] | None = None
    artifact: dict[str, Any] | None = None
    citation_ids: list[str] | None = None
    final_state: dict[str, Any] | None = None


class Observations(StrictModel):
    schema_version: Literal["task-observations/v1"]
    suite_id: str
    suite_version: str
    suite_hash: str
    run_id: str = Field(min_length=1)
    producer_name: str = Field(min_length=1)
    producer_version: str = Field(min_length=1)
    tasks: list[Observation]

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("observations task_id values must be unique")
        return self


class TaskResult(StrictModel):
    task_id: str
    scenario: str
    status: Literal["success", "failed", "incomplete"]
    diagnostics: list[str]
    checks: dict[str, bool | None]
    expected: Expected
    observed: Observation | None


class Evaluation(StrictModel):
    schema_version: Literal["task-result/v1"] = "task-result/v1"
    suite_id: str
    suite_version: str
    suite_hash: str
    source_kind: str
    run_id: str
    producer_name: str
    producer_version: str
    summary: dict[str, float | int]
    tasks: list[TaskResult]


def lock_path(suite_path: Path) -> Path:
    return suite_path.with_suffix(".lock.json")


def load_suite(path: Path) -> Suite:
    suite = Suite.model_validate_json(path.read_text(encoding="utf-8-sig"))
    lock = json.loads(lock_path(path).read_text(encoding="utf-8-sig"))
    if lock != {"schema_version": "task-suite-lock/v1", "suite_hash": suite.content_hash()}:
        raise ValueError("suite lock does not match suite content")
    return suite


def load_observations(path: Path, suite: Suite) -> Observations:
    observations = Observations.model_validate_json(path.read_text(encoding="utf-8-sig"))
    _validate_binding(suite, observations)
    return observations


def _validate_binding(suite: Suite, observations: Observations) -> None:
    if (observations.suite_id, observations.suite_version, observations.suite_hash) != (
        suite.suite_id, suite.suite_version, suite.content_hash()
    ):
        raise ValueError("observations suite id, version, or hash does not match")
    unknown = {item.task_id for item in observations.tasks} - {task.task_id for task in suite.tasks}
    if unknown:
        raise ValueError(f"unknown observation task_id: {', '.join(sorted(unknown))}")


def _json_equal(expected: Any, actual: Any) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _json_equal(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            _json_equal(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return bool(expected == actual)


def _matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Expected fields are assertions; extra observed fields are permitted."""
    return all(
        key in observed and _json_equal(value, observed[key])
        for key, value in expected.items()
    )


def _score_task(task: Task, observed: Observation | None) -> TaskResult:
    if observed is None or any(
        value is None
        for value in (
            observed.retrieved_evidence_ids,
            observed.artifact,
            observed.citation_ids,
            observed.final_state,
        )
    ):
        return TaskResult(
            task_id=task.task_id,
            scenario=task.scenario,
            status="incomplete",
            diagnostics=["incomplete_observation"],
            checks={name: None for name in ("retrieval", "artifact", "citations", "state")},
            expected=task.expected,
            observed=observed,
        )
    assert observed.retrieved_evidence_ids is not None
    assert observed.artifact is not None
    assert observed.citation_ids is not None
    assert observed.final_state is not None
    retrieved = set(observed.retrieved_evidence_ids)
    cited = set(observed.citation_ids)
    authoritative = {item.evidence_id for item in task.evidence if item.role == "authoritative"}
    conflicting = {item.evidence_id for item in task.evidence if item.role == "conflicting"}
    expected_citations = set(task.expected.citation_ids)
    checks: dict[str, bool | None] = {
        "retrieval": expected_citations <= retrieved,
        "artifact": _matches(task.expected.artifact, observed.artifact),
        "citations": expected_citations <= cited and cited <= (retrieved & authoritative),
        "state": _matches(task.expected.final_state, observed.final_state),
    }
    diagnostics: list[str] = []
    if not checks["retrieval"]:
        diagnostics.append("retrieval_missing_evidence")
    if not checks["artifact"]:
        if any(key not in observed.artifact for key in task.expected.artifact):
            diagnostics.append("missing_required_field")
        elif cited & conflicting:
            diagnostics.append("knowledge_conflict")
        elif checks["retrieval"]:
            diagnostics.append("reasoning_or_selection_error")
        else:
            diagnostics.append("undetermined")
    if not checks["citations"]:
        diagnostics.append("invalid_citation")
    if not checks["state"]:
        diagnostics.append("state_mismatch")
    return TaskResult(
        task_id=task.task_id,
        scenario=task.scenario,
        status="success" if all(checks.values()) else "failed",
        diagnostics=diagnostics,
        checks=checks,
        expected=task.expected,
        observed=observed,
    )


def evaluate(suite: Suite, observations: Observations) -> Evaluation:
    _validate_binding(suite, observations)
    by_id = {item.task_id: item for item in observations.tasks}
    tasks = [_score_task(task, by_id.get(task.task_id)) for task in suite.tasks]
    successful = sum(task.status == "success" for task in tasks)
    received = len(observations.tasks)
    complete = sum(task.status != "incomplete" for task in tasks)
    total = len(suite.tasks)
    checks = [value for task in tasks for value in task.checks.values() if value is not None]
    counts = Counter(tag for task in tasks for tag in task.diagnostics)
    summary: dict[str, float | int] = {
        "total_tasks": total,
        "successful_tasks": successful,
        "received_observations": received,
        "complete_observations": complete,
        "incomplete_tasks": sum(task.status == "incomplete" for task in tasks),
        "task_success_rate": successful / total,
        "observation_coverage": complete / total,
        "received_observation_coverage": received / total,
        "observed_success_rate": successful / complete if complete else 0.0,
        "step_pass_rate": sum(checks) / len(checks) if checks else 0.0,
    }
    summary.update({f"diagnostic_{tag}": count for tag, count in sorted(counts.items())})
    return Evaluation(
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.content_hash(),
        source_kind=suite.source_kind,
        run_id=observations.run_id,
        producer_name=observations.producer_name,
        producer_version=observations.producer_version,
        summary=summary,
        tasks=tasks,
    )


def evaluate_files(suite_path: Path, observations_path: Path) -> Evaluation:
    suite = load_suite(suite_path)
    return evaluate(suite, load_observations(observations_path, suite))


def compare(baseline: Evaluation, candidate: Evaluation) -> dict[str, Any]:
    if (baseline.suite_id, baseline.suite_version, baseline.suite_hash) != (
        candidate.suite_id, candidate.suite_version, candidate.suite_hash
    ):
        raise ValueError("task comparisons require the same frozen suite")
    if [task.task_id for task in baseline.tasks] != [task.task_id for task in candidate.tasks]:
        raise ValueError("task comparisons require the same task IDs and order")
    changes = [
        {
            "task_id": left.task_id,
            "before": {"status": left.status, "checks": left.checks,
                       "diagnostics": left.diagnostics},
            "after": {"status": right.status, "checks": right.checks,
                      "diagnostics": right.diagnostics},
        }
        for left, right in zip(baseline.tasks, candidate.tasks, strict=True)
        if (left.status, left.checks, left.diagnostics)
        != (right.status, right.checks, right.diagnostics)
    ]
    return {
        "suite_hash": baseline.suite_hash,
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "success_rate_delta": (
            float(candidate.summary["task_success_rate"])
            - float(baseline.summary["task_success_rate"])
        ),
        "changes": changes,
    }


def load_result(path: Path) -> Evaluation:
    return Evaluation.model_validate_json(path.read_text(encoding="utf-8-sig"))
