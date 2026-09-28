"""The workbench's report bytes must match what ``rag-quality report`` writes.

The web app calls the same :func:`generate_reports` the CLI calls with the
same inputs; this test is the contract that keeps it that way. If someone
changes the web report handler to build its own payload instead of reusing
the reporting module, this test catches the divergence.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from rag_quality_lab.web.app import create_app
from tests.web.conftest import write_mock_workbench_inputs
from tests.web.wsgi_client import call


def test_web_report_matches_cli_report(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    database_path = tmp_path / "experiments.sqlite3"

    run_result = subprocess.run(
        [sys.executable, "-m", "rag_quality_lab.cli", "run", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
    experiment_id = json.loads(run_result.stdout)["experiment_id"]

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

    web_output_dir = tmp_path / "web-report"
    app = create_app(allow_live=False, workspace=tmp_path)
    web_report = call(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/report",
        query=f"database={database_path}&output={web_output_dir}",
    )
    assert web_report.status_code == 200
    web_body = web_report.json()

    cli_json_path = cli_output_dir / f"{experiment_id}.json"
    cli_html_path = cli_output_dir / f"{experiment_id}.html"
    web_json_path = Path(web_body["report_json"])
    web_html_path = Path(web_body["report_html"])

    assert cli_json_path.read_bytes() == web_json_path.read_bytes()
    assert cli_html_path.read_bytes() == web_html_path.read_bytes()
    assert json.loads(cli_report.stdout)["json_sha256"] == web_body["json_sha256"]
    assert json.loads(cli_report.stdout)["html_sha256"] == web_body["html_sha256"]


def test_started_experiment_report_matches_cli_regeneration(tmp_path: Path) -> None:
    """A run started from the workbench produces the same report the CLI
    would regenerate for it afterwards, not just runs started by the CLI."""

    config_path = write_mock_workbench_inputs(tmp_path)
    database_path = tmp_path / "experiments.sqlite3"
    app = create_app(allow_live=False, workspace=tmp_path)

    start_response = call(
        app, "POST", "/api/experiments/start", json_body={"config": str(config_path)}
    )
    token = start_response.json()["token"]
    deadline = time.monotonic() + 10
    run_state = {}
    while time.monotonic() < deadline:
        run_state = call(app, "GET", f"/api/runs/{token}").json()
        if run_state["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert run_state["status"] == "completed"
    experiment_id = run_state["experiment_id"]

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

    web_output_dir = tmp_path / "web-report"
    web_report = call(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/report",
        query=f"database={database_path}&output={web_output_dir}",
    )
    assert web_report.status_code == 200
    web_body = web_report.json()
    assert (cli_output_dir / f"{experiment_id}.json").read_bytes() == Path(
        web_body["report_json"]
    ).read_bytes()
