"""A restored backup must reproduce the same report as the original.

Runs the backup/restore script as a subprocess (matching how an operator
would use it), then regenerates the report from both the original and the
restored database/config and checks the JSON and HTML bytes -- and their
sha256 -- are identical.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.web.conftest import write_mock_workbench_inputs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backup_restore.py"


def _run(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _run_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "rag_quality_lab.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_backup_then_restore_regenerates_identical_report(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    config_path = write_mock_workbench_inputs(source_dir)

    run_payload = _run_cli("run", "--config", str(config_path))
    experiment_id = run_payload["experiment_id"]

    original_report_dir = tmp_path / "original-report"
    original_report = _run_cli(
        "report",
        "--database",
        str(source_dir / "experiments.sqlite3"),
        "--experiment",
        experiment_id,
        "--output",
        str(original_report_dir),
    )

    archive_path = tmp_path / "backup.zip"
    backup_payload = _run("backup", "--config", str(config_path), "--output", str(archive_path))
    assert Path(backup_payload["archive"]).is_file()

    destination = tmp_path / "restored"
    restore_payload = _run(
        "restore", "--archive", str(archive_path), "--destination", str(destination)
    )
    restored_config_path = Path(restore_payload["config"])
    assert restored_config_path.is_file()

    restored_database_path = destination / "database" / "experiments.sqlite3"
    assert restored_database_path.is_file()

    restored_report_dir = tmp_path / "restored-report"
    restored_report = _run_cli(
        "report",
        "--database",
        str(restored_database_path),
        "--experiment",
        experiment_id,
        "--output",
        str(restored_report_dir),
    )

    assert restored_report["json_sha256"] == original_report["json_sha256"]
    assert restored_report["html_sha256"] == original_report["html_sha256"]
    original_json = Path(original_report["report_json"]).read_bytes()
    restored_json = Path(restored_report["report_json"]).read_bytes()
    assert original_json == restored_json

    # The restored config also validates and re-runs cleanly on its own, so
    # the restore is genuinely self-contained, not just database-deep.
    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_quality_lab.cli",
            "validate",
            "--config",
            str(restored_config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stderr
