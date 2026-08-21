from pathlib import Path

import pytest

from rag_quality_lab.metrics.calibration import (
    HumanJudgePair,
    JudgeSample,
    calibrate,
    export_blind_annotations,
    import_human_annotations,
)


def test_calibration_requires_twelve_labels() -> None:
    result = calibrate([HumanJudgePair(human_score=4, judge_score=4)] * 11)

    assert not result.blocking_eligible
    assert result.reason == "at least 12 labels are required"


def test_empty_calibration_has_defined_zero_metrics() -> None:
    result = calibrate([])

    assert result.exact_agreement == 0.0
    assert result.within_one_rate == 0.0
    assert result.mean_absolute_error == 0.0


def test_calibrated_judge_can_block() -> None:
    pairs = [HumanJudgePair(human_score=4, judge_score=4)] * 10 + [
        HumanJudgePair(human_score=3, judge_score=4),
        HumanJudgePair(human_score=5, judge_score=4),
    ]

    result = calibrate(pairs)

    assert result.exact_agreement == pytest.approx(10 / 12)
    assert result.within_one_rate == 1.0
    assert result.mean_absolute_error <= 1.0
    assert result.blocking_eligible


def test_calibration_below_agreement_threshold_is_diagnostic() -> None:
    pairs = [HumanJudgePair(human_score=1, judge_score=5)] * 12

    result = calibrate(pairs)

    assert not result.blocking_eligible
    assert result.reason == "agreement thresholds not met"


def test_blind_annotation_export_removes_model_and_judge_fields(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    export_blind_annotations(
        [
            JudgeSample(
                sample_id="8f0dbdb739676f484782f8d6",
                question="Q",
                reference_answer="R",
                candidate_answer="A",
                evidence=["E"],
                model="secret-model",
                config_id="config-a",
                judge_score=4,
            )
        ],
        path,
    )

    exported = path.read_text(encoding="utf-8")
    assert "secret-model" not in exported
    assert "config-a" not in exported
    assert "judge_score" not in exported
    assert '"sample_id":"8f0dbdb739676f484782f8d6"' in exported
    assert '"content_hash":' in exported

    path.write_text(exported.replace("null", "4"), encoding="utf-8")
    annotations = import_human_annotations(path)
    assert annotations[0].human_score == 4
    assert len(annotations[0].content_hash) == 64


def test_human_annotation_import_rejects_missing_and_duplicate_scores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.jsonl"
    export_blind_annotations(
        [
            JudgeSample(
                sample_id="8f0dbdb739676f484782f8d6",
                question="Q",
                reference_answer="R",
                candidate_answer="A",
                evidence=["E"],
                model="model",
                config_id="config",
                judge_score=4,
            )
        ],
        path,
    )

    with pytest.raises(ValueError, match="human_score"):
        import_human_annotations(path)

    completed = path.read_text(encoding="utf-8").replace("null", "4")
    path.write_text(f"\n{completed}{completed}", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        import_human_annotations(path)


def test_human_annotation_import_rejects_modified_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    export_blind_annotations(
        [
            JudgeSample(
                sample_id="8f0dbdb739676f484782f8d6",
                question="Q",
                reference_answer="R",
                candidate_answer="A",
                evidence=["E"],
                model="model",
                config_id="config",
                judge_score=4,
            )
        ],
        path,
    )
    payload = (
        path.read_text(encoding="utf-8")
        .replace('"candidate_answer":"A"', '"candidate_answer":"changed"')
        .replace("null", "4")
    )
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="content hash"):
        import_human_annotations(path)
