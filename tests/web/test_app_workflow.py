"""End-to-end workbench workflow: precheck, run, progress, results, report."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from rag_quality_lab.web.app import WSGIApp, create_app
from tests.web.conftest import write_mock_workbench_inputs
from tests.web.wsgi_client import call


def _run_to_completion(
    app: WSGIApp, token: str, *, timeout_seconds: float = 10.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = call(app, "GET", f"/api/runs/{token}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return cast(dict[str, Any], body)
        time.sleep(0.02)
    raise AssertionError(f"run {token} did not finish within {timeout_seconds}s")


def test_full_workbench_workflow(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    dataset_response = call(
        app, "GET", "/api/dataset", query=f"dataset={tmp_path / 'dataset.json'}"
    )
    assert dataset_response.status_code == 200
    assert "splits" in dataset_response.json() or "dev" in dataset_response.json()

    config_response = call(app, "GET", "/api/config", query=f"config={config_path}")
    assert config_response.status_code == 200
    config_body = config_response.json()
    assert config_body["mode"] == "mock"
    assert len(config_body["arms"]) == 2

    precheck_response = call(app, "GET", "/api/precheck", query=f"config={config_path}")
    assert precheck_response.status_code == 200
    precheck_body = precheck_response.json()
    assert precheck_body["mode"] == "mock"
    assert precheck_body["total_request_count"] > 0

    start_response = call(
        app, "POST", "/api/experiments/start", json_body={"config": str(config_path)}
    )
    assert start_response.status_code == 200
    token = start_response.json()["token"]

    run_state = _run_to_completion(app, token)
    assert run_state["status"] == "completed"
    experiment_id = run_state["experiment_id"]
    assert experiment_id is not None

    database_path = tmp_path / "experiments.sqlite3"
    listing = call(app, "GET", "/api/experiments", query=f"database={database_path}")
    assert listing.status_code == 200
    assert any(entry["id"] == experiment_id for entry in listing.json()["experiments"])

    detail = call(
        app,
        "GET",
        f"/api/experiments/{experiment_id}",
        query=f"database={database_path}",
    )
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["status"] == "completed"
    assert detail_body["progress"]["done"] == detail_body["progress"]["total"]
    assert detail_body["failures"] == []

    results = call(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/results",
        query=f"database={database_path}&limit=2&offset=0",
    )
    assert results.status_code == 200
    results_body = results.json()
    assert results_body["total"] == detail_body["case_result_count"]
    assert len(results_body["results"]) == 2
    first_result = results_body["results"][0]
    assert first_result["question"]
    assert first_result["evidence"]

    output_dir = tmp_path / "reports"
    report = call(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/report",
        json_body={"database": str(database_path), "output": str(output_dir)},
    )
    assert report.status_code == 200
    report_body = report.json()
    assert Path(report_body["report_json"]).is_file()
    assert Path(report_body["report_html"]).is_file()

    download = call(
        app,
        "GET",
        "/api/artifacts/download",
        query=f"output={output_dir}&file={experiment_id}.json",
    )
    assert download.status_code == 200
    assert download.body == Path(report_body["report_json"]).read_bytes()


def test_cancel_refuses_after_completion(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)
    start_response = call(
        app, "POST", "/api/experiments/start", json_body={"config": str(config_path)}
    )
    token = start_response.json()["token"]
    run_state = _run_to_completion(app, token)
    experiment_id = run_state["experiment_id"]
    database_path = tmp_path / "experiments.sqlite3"

    cancel_response = call(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/cancel",
        json_body={"database": str(database_path), "experiment_id": experiment_id},
    )
    # A completed experiment cannot be cancelled; the store rejects it and the
    # web layer surfaces that as a 400, not a silent success.
    assert cancel_response.status_code == 400


def test_live_run_refused_without_allow_live(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        "provider: local-test\n"
        "currency: CNY\n"
        "verified_at: 2026-01-01\n"
        "source_url: https://example.invalid/pricing\n"
        "models: {}\n",
        encoding="utf-8",
    )
    config_text = config_path.read_text(encoding="utf-8").replace(
        "mode: mock", f"mode: live\npricing_path: {pricing_path}"
    )
    config_path.write_text(config_text, encoding="utf-8")
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(
        app,
        "POST",
        "/api/experiments/start",
        json_body={"config": str(config_path), "confirm_live_run": True},
    )
    assert response.status_code == 403
