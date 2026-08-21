import json
import subprocess
import sys
from pathlib import Path


def test_readme_offline_command_succeeds(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_quality_lab.cli",
            "run",
            "--config",
            "configs/offline.yaml",
            "--artifact-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert Path(payload["report_json"]).parent == tmp_path
    assert Path(payload["report_html"]).parent == tmp_path
