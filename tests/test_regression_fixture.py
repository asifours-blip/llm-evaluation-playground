import json
import subprocess
import sys
from pathlib import Path


def test_offline_regression_fixture_exercises_real_gate() -> None:
    fixture = Path("tests/fixtures/offline_baseline.json")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_quality_lab.cli",
            "regression",
            "--fixture",
            str(fixture),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "failed_metrics": [],
        "passed": True,
        "skipped_metrics": [],
    }
