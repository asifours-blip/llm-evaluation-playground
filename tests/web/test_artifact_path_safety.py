"""Report downloads must never escape their artifact directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_quality_lab.web.service import PathEscapesArtifactDir, artifact_download_path


def test_plain_file_resolves(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    resolved = artifact_download_path(tmp_path, "report.json")
    assert resolved == (tmp_path / "report.json").resolve()


def test_nested_file_resolves(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "report.json").write_text("{}", encoding="utf-8")
    resolved = artifact_download_path(tmp_path, "nested/report.json")
    assert resolved == (tmp_path / "nested" / "report.json").resolve()


@pytest.mark.parametrize(
    "requested",
    [
        "../secret.txt",
        "../../etc/passwd",
        "nested/../../secret.txt",
        "/etc/passwd",
    ],
)
def test_traversal_is_refused(tmp_path: Path, requested: str) -> None:
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    try:
        with pytest.raises(PathEscapesArtifactDir):
            artifact_download_path(tmp_path, requested)
    finally:
        secret.unlink(missing_ok=True)


def test_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        artifact_download_path(tmp_path, "missing.json")
