"""``report`` and ``compare`` write files, so they must be POST-only.

A GET must be refused with 405 (not silently accepted or 404'd, which would
suggest the route doesn't exist), a cross-origin POST must be refused with
403, and a normal same-origin POST must still work and match what the CLI
writes for the same experiment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rag_quality_lab.web.app import create_app
from tests.web.conftest import write_mock_workbench_inputs
from tests.web.wsgi_client import call


def _run_experiment(config_path: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "rag_quality_lab.cli", "run", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    experiment_id: str = json.loads(result.stdout)["experiment_id"]
    return experiment_id


def test_get_on_report_is_refused_with_405(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    database_path = tmp_path / "experiments.sqlite3"
    experiment_id = _run_experiment(config_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/report",
        query=f"database={database_path}&output={tmp_path / 'reports'}",
    )
    assert response.status_code == 405


def test_get_on_compare_is_refused_with_405(tmp_path: Path) -> None:
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(
        app,
        "GET",
        "/api/compare",
        query=(
            f"database={tmp_path / 'experiments.sqlite3'}&baseline=a&candidate=b"
            f"&baseline_config=c&candidate_config=d&output={tmp_path / 'reports'}"
        ),
    )
    assert response.status_code == 405


def test_cross_origin_post_to_report_is_refused_with_403(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    database_path = tmp_path / "experiments.sqlite3"
    experiment_id = _run_experiment(config_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/report",
        json_body={"database": str(database_path), "output": str(tmp_path / "reports")},
        origin="http://evil.example",
    )
    assert response.status_code == 403


def test_cross_origin_post_to_compare_is_refused_with_403(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    database_path = tmp_path / "experiments.sqlite3"
    _run_experiment(config_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(
        app,
        "POST",
        "/api/compare",
        json_body={
            "database": str(database_path),
            "baseline": "latest-live",
            "candidate": "latest-live",
            "baseline_config": "chunk300-overlap50-top2-direct",
            "candidate_config": "chunk300-overlap50-top2-evidence_first",
            "output": str(tmp_path / "reports"),
        },
        origin="http://evil.example",
    )
    assert response.status_code == 403


def test_same_origin_post_generates_a_report_matching_the_cli(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    database_path = tmp_path / "experiments.sqlite3"
    experiment_id = _run_experiment(config_path)

    cli_output_dir = tmp_path / "cli-report"
    cli_report = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_quality_lab.cli",
            "report",
            "--database",
            str(database_path),
            "--experiment",
            experiment_id,
            "--output",
            str(cli_output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli_report.returncode == 0, cli_report.stderr

    app = create_app(allow_live=False, workspace=tmp_path)
    web_output_dir = tmp_path / "web-report"
    for origin in (None, "http://127.0.0.1:8765", "http://localhost:8765"):
        response = call(
            app,
            "POST",
            f"/api/experiments/{experiment_id}/report",
            json_body={"database": str(database_path), "output": str(web_output_dir)},
            origin=origin,
        )
        assert response.status_code == 200, response.body
        web_body = response.json()
        assert Path(web_body["report_json"]).read_bytes() == (
            cli_output_dir / f"{experiment_id}.json"
        ).read_bytes()
        assert web_body["json_sha256"] == json.loads(cli_report.stdout)["json_sha256"]
