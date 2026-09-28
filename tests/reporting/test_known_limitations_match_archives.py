"""KNOWN_LIMITATIONS' archived judge numbers must match the evidence files.

Guards against the limitations text drifting from, or transposing, the
figures actually recorded in docs/artifacts/*/evidence-summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from rag_quality_lab.reporting.report import KNOWN_LIMITATIONS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVE_384_EVIDENCE = (
    REPOSITORY_ROOT / "docs" / "artifacts" / "live-384-2026-08-21" / "evidence-summary.json"
)
HTTP_INSTRUMENTED_EVIDENCE = (
    REPOSITORY_ROOT
    / "docs"
    / "artifacts"
    / "http-instrumented-2026-08-21"
    / "evidence-summary.json"
)


def _quality(path: Path) -> dict[str, float]:
    return cast(dict[str, float], json.loads(path.read_text(encoding="utf-8"))["quality"])


def _limitations_text() -> str:
    return " ".join(KNOWN_LIMITATIONS)


def test_live_384_numbers_match_archive() -> None:
    quality = _quality(LIVE_384_EVIDENCE)
    text = _limitations_text()
    assert f"judge_pass_rate ~{round(quality['judge_pass_rate'], 3)}" in text
    assert f"false_answer_rate ~{round(quality['false_answer_rate'], 3)}" in text


def test_http_instrumented_numbers_match_archive() -> None:
    quality = _quality(HTTP_INSTRUMENTED_EVIDENCE)
    text = _limitations_text()
    assert f"judge_pass_rate ~{round(quality['judge_pass_rate'], 3)}" in text
    assert f"false_answer_rate ~{round(quality['false_answer_rate'], 3)}" in text


def test_the_two_runs_numbers_are_not_swapped_or_conflated() -> None:
    live_384 = _quality(LIVE_384_EVIDENCE)
    http_instrumented = _quality(HTTP_INSTRUMENTED_EVIDENCE)
    # The two archived runs have distinct figures; if the limitations text
    # only quoted one run's numbers (or averaged/merged them), one of the
    # two tests above would already fail. This assertion documents why two
    # separate checks matter instead of one shared one.
    assert round(live_384["judge_pass_rate"], 3) != round(
        http_instrumented["judge_pass_rate"], 3
    )
