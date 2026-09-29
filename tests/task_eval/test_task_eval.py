"""Complete task scoring cannot hide missing work or trust imported expectations."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import pytest
from pydantic import ValidationError

from rag_quality_lab.task_eval.core import (
    Observation,
    Observations,
    Suite,
    compare,
    evaluate,
    evaluate_files,
    load_observations,
    load_suite,
)
from rag_quality_lab.task_eval.report import write_report
from rag_quality_lab.web.app import create_app
from tests.web.wsgi_client import call

DATA = Path(__file__).parents[2] / "data" / "task_eval"


def test_frozen_demo_scores_all_twelve_tasks() -> None:
    result = evaluate_files(DATA / "suite.json", DATA / "observations-demo.json")
    assert result.source_kind == "synthetic_demo"
    assert result.summary["total_tasks"] == 12
    assert result.summary["successful_tasks"] == 4
    assert result.summary["task_success_rate"] == 4 / 12
    assert result.summary["received_observations"] == 11
    assert result.summary["complete_observations"] == 10
    assert result.summary["observation_coverage"] == 10 / 12
    assert result.summary["observed_success_rate"] == 4 / 10
    assert result.tasks[10].status == "incomplete"  # absent task remains in denominator
    assert result.tasks[11].status == "incomplete"  # partial task remains in denominator
    assert "retrieval_missing_evidence" in result.tasks[4].diagnostics
    assert "reasoning_or_selection_error" in result.tasks[5].diagnostics
    assert "knowledge_conflict" in result.tasks[6].diagnostics
    assert "missing_required_field" in result.tasks[7].diagnostics
    assert "state_mismatch" in result.tasks[8].diagnostics
    assert "invalid_citation" in result.tasks[9].diagnostics


def test_bad_binding_and_untrusted_expectations_are_refused(tmp_path: Path) -> None:
    suite = load_suite(DATA / "suite.json")
    original = json.loads((DATA / "observations-demo.json").read_text(encoding="utf-8"))
    variants = []
    wrong_hash = json.loads(json.dumps(original))
    wrong_hash["suite_hash"] = "wrong"
    variants.append(wrong_hash)
    duplicate = json.loads(json.dumps(original))
    duplicate["tasks"].append(duplicate["tasks"][0])
    variants.append(duplicate)
    unknown = json.loads(json.dumps(original))
    unknown["tasks"][0]["task_id"] = "not-in-suite"
    variants.append(unknown)
    forged_expected = json.loads(json.dumps(original))
    forged_expected["tasks"][0]["expected"] = {"artifact": {}}
    variants.append(forged_expected)
    for index, payload in enumerate(variants):
        path = tmp_path / f"bad-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises((ValueError, ValidationError)):
            load_observations(path, suite)


def test_public_evaluate_checks_binding_and_strict_json_types() -> None:
    suite = load_suite(DATA / "suite.json")
    observations = load_observations(DATA / "observations-demo.json", suite)
    with pytest.raises(ValueError, match="hash"):
        evaluate(suite, observations.model_copy(update={"suite_hash": "forged"}))
    first = observations.tasks[0].model_copy(deep=True)
    first.artifact["action"] = True  # type: ignore[index]
    changed = observations.model_copy(update={"tasks": [first, *observations.tasks[1:]]})
    assert evaluate(suite, changed).tasks[0].status == "failed"
    minimal = Suite.model_validate({
        "schema_version": "task-eval/v1", "suite_id": "small", "suite_version": "1",
        "source_kind": "synthetic_demo", "tasks": [suite.tasks[0].model_dump(mode="json")],
    })
    assert len(minimal.tasks) == 1


def test_compare_requires_same_suite_and_reports_task_changes() -> None:
    suite = load_suite(DATA / "suite.json")
    baseline_obs = load_observations(DATA / "observations-demo.json", suite)
    baseline = evaluate(suite, baseline_obs)
    replacement = Observation(
        task_id=suite.tasks[4].task_id,
        retrieved_evidence_ids=["K05"],
        artifact=suite.tasks[4].expected.artifact,
        citation_ids=["K05"],
        final_state=suite.tasks[4].expected.final_state,
    )
    tasks = [
        replacement if task.task_id == replacement.task_id else task
        for task in baseline_obs.tasks
    ]
    candidate = evaluate(
        suite, baseline_obs.model_copy(update={"run_id": "candidate", "tasks": tasks})
    )
    delta = compare(baseline, candidate)
    assert delta["success_rate_delta"] == pytest.approx(1 / 12)
    assert {change["task_id"] for change in delta["changes"]} == {replacement.task_id}
    still_failed = baseline_obs.tasks[4].model_copy(deep=True)
    still_failed.retrieved_evidence_ids = ["K05"]
    still_failed.citation_ids = ["K05"]
    still_failed.artifact = {"action": "wrong", "message_code": "wrong"}
    changed_tasks = [
        still_failed if task.task_id == still_failed.task_id else task
        for task in baseline_obs.tasks
    ]
    changed_failure = evaluate(
        suite, baseline_obs.model_copy(update={"run_id": "changed", "tasks": changed_tasks})
    )
    changed = compare(baseline, changed_failure)["changes"]
    assert any(item["task_id"] == still_failed.task_id for item in changed)
    assert baseline.tasks[4].status == changed_failure.tasks[4].status == "failed"
    with pytest.raises(ValueError, match="same frozen suite"):
        compare(baseline, candidate.model_copy(update={"suite_hash": "different"}))


def test_report_is_deterministic_and_web_is_read_only_and_confined(tmp_path: Path) -> None:
    result = evaluate_files(DATA / "suite.json", DATA / "observations-demo.json")
    json_path, html_path = write_report(result, tmp_path)
    first_json, first_html = json_path.read_bytes(), html_path.read_bytes()
    write_report(result, tmp_path)
    assert json_path.read_bytes() == first_json
    assert html_path.read_bytes() == first_html
    assert b"4 / 12" in first_html
    app = create_app(workspace=tmp_path)
    query = urlencode({"report": str(json_path)})
    view = call(app, "GET", "/api/task-eval/result", query=query)
    assert view.status_code == 200
    assert view.json() == json.loads(first_json)
    compare_query = urlencode({"baseline": str(json_path), "candidate": str(json_path)})
    comparison = call(app, "GET", "/api/task-eval/compare", query=compare_query)
    assert comparison.status_code == 200
    assert comparison.json()["success_rate_delta"] == 0
    escaped = call(
        app, "GET", "/api/task-eval/result",
        query=urlencode({"report": str(DATA / "suite.json")}),
    )
    assert escaped.status_code == 400
    assert call(app, "POST", "/api/task-eval/result", json_body={}).status_code == 405
    assert json_path.read_bytes() == first_json


def test_html_report_escapes_untrusted_run_and_scenario(tmp_path: Path) -> None:
    result = evaluate_files(DATA / "suite.json", DATA / "observations-demo.json")
    result = result.model_copy(deep=True)
    result.run_id = '<img src=x onerror=alert(1)>'
    result.tasks[0].scenario = '<script>alert(1)</script>'
    _, html_path = write_report(result, tmp_path)
    html = html_path.read_text(encoding="utf-8")
    assert "<img src=x" not in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;img" in html
    assert "&lt;script&gt;" in html


def test_boolean_assertion_does_not_accept_numeric_zero() -> None:
    base = load_suite(DATA / "suite.json")
    payload = base.model_dump(mode="json")
    payload["tasks"] = [payload["tasks"][0]]
    payload["tasks"][0]["expected"]["artifact"] = {"approved": False}
    suite = Suite.model_validate(payload)
    observed = {
        "schema_version": "task-observations/v1",
        "suite_id": suite.suite_id,
        "suite_version": suite.suite_version,
        "suite_hash": suite.content_hash(),
        "run_id": "strict-type",
        "producer_name": "test",
        "producer_version": "1",
        "tasks": [{
            "task_id": suite.tasks[0].task_id,
            "retrieved_evidence_ids": ["K01"],
            "artifact": {"approved": 0},
            "citation_ids": ["K01"],
            "final_state": suite.tasks[0].expected.final_state,
        }],
    }
    result = evaluate(suite, Observations.model_validate(observed))
    assert result.tasks[0].status == "failed"
    assert result.tasks[0].checks["artifact"] is False
