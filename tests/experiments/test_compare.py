import pytest
from pydantic import ValidationError

from rag_quality_lab.domain.models import (
    CaseResult,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
)
from rag_quality_lab.experiments.compare import (
    ComparisonResult,
    MetricDelta,
    RegressionRule,
    compare_experiments,
    evaluate_regression,
)
from rag_quality_lab.metrics.calibration import CalibrationResult


def identity(name: str) -> ExperimentIdentity:
    return ExperimentIdentity(
        name=name,
        mode="mock",
        commit_sha="abc",
        dirty=False,
        dataset_hash="dataset",
        prompt_hashes={"direct": "prompt"},
        config={},
        random_seed=42,
        python_version="3.11",
    )


def case_result(case_id: str, answer_f1: float) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        config_id="config",
        model="model",
        metrics={
            "answer_f1": answer_f1,
            "retrieval_recall_at_k": 1.0,
            "false_answer": 0.0,
            "over_abstention": 0.0,
        },
        status="completed",
    )


def experiment(
    experiment_id: str,
    recall: float,
    answer_f1: float,
    *,
    false_answer_rate: float = 0.0,
) -> ExperimentRecord:
    return ExperimentRecord(
        id=experiment_id,
        identity=identity(experiment_id),
        status=ExperimentStatus.COMPLETED,
        case_results=[case_result("rag-007", answer_f1)],
        summary={
            "retrieval_recall_at_k": recall,
            "answer_f1": answer_f1,
            "false_answer_rate": false_answer_rate,
            "failure_count": 0.0,
        },
    )


def test_comparison_reports_metric_and_case_deltas() -> None:
    comparison = compare_experiments(
        experiment("baseline", 0.8, 1.0),
        experiment("candidate", 0.7, 0.5),
    )

    assert comparison.metric_deltas["retrieval_recall_at_k"].absolute == -0.1
    assert comparison.metric_deltas["retrieval_recall_at_k"].percentage == -12.5
    assert comparison.regressed_case_ids == ["rag-007"]


def test_comparison_marks_percentage_unknown_for_zero_baseline() -> None:
    comparison = compare_experiments(
        experiment("baseline", 0.0, 1.0),
        experiment("candidate", 0.1, 1.0),
    )

    assert comparison.metric_deltas["retrieval_recall_at_k"].percentage is None


def test_uncalibrated_judge_metric_cannot_fail_gate() -> None:
    comparison = ComparisonResult(
        baseline_id="base",
        candidate_id="candidate",
        metric_deltas={
            "judge_correctness": MetricDelta(
                baseline=0.9,
                candidate=0.5,
                absolute=-0.4,
                percentage=-44.444,
            )
        },
    )
    calibration = CalibrationResult(
        label_count=0,
        exact_agreement=0,
        within_one_rate=0,
        mean_absolute_error=0,
        blocking_eligible=False,
        reason="at least 12 labels are required",
    )

    verdict = evaluate_regression(
        comparison,
        rules=[RegressionRule(metric="judge_correctness", minimum_delta=0)],
        judge_calibration=calibration,
    )

    assert verdict.passed
    assert verdict.skipped_metrics == ["judge_correctness"]


def test_lower_is_better_metric_uses_maximum_delta() -> None:
    comparison = compare_experiments(
        experiment("baseline", 0.8, 1.0, false_answer_rate=0.0),
        experiment("candidate", 0.8, 1.0, false_answer_rate=0.1),
    )

    verdict = evaluate_regression(
        comparison,
        rules=[RegressionRule(metric="false_answer_rate", maximum_delta=0)],
    )

    assert not verdict.passed
    assert verdict.failed_metrics == ["false_answer_rate"]


def test_regression_rule_requires_at_least_one_boundary() -> None:
    with pytest.raises(ValidationError):
        RegressionRule(metric="answer_f1")


def test_comparison_reports_improved_and_missing_candidate_cases() -> None:
    baseline = experiment("baseline", 0.8, 0.5)
    improved = experiment("improved", 0.8, 1.0)
    missing = experiment("missing", 0.8, 1.0).model_copy(
        update={"case_results": []}
    )

    assert compare_experiments(baseline, improved).improved_case_ids == ["rag-007"]
    assert compare_experiments(baseline, missing).regressed_case_ids == ["rag-007"]


def test_gate_fails_minimum_and_missing_metrics() -> None:
    comparison = compare_experiments(
        experiment("baseline", 0.8, 1.0),
        experiment("candidate", 0.7, 1.0),
    )

    verdict = evaluate_regression(
        comparison,
        rules=[
            RegressionRule(metric="retrieval_recall_at_k", minimum_delta=0),
            RegressionRule(metric="missing_metric", minimum_delta=0),
        ],
    )

    assert verdict.failed_metrics == ["retrieval_recall_at_k", "missing_metric"]


def test_calibrated_judge_metric_can_fail_gate() -> None:
    comparison = ComparisonResult(
        baseline_id="base",
        candidate_id="candidate",
        metric_deltas={
            "judge_correctness": MetricDelta(
                baseline=0.9,
                candidate=0.5,
                absolute=-0.4,
                percentage=-44.444,
            )
        },
    )
    calibration = CalibrationResult(
        label_count=12,
        exact_agreement=1,
        within_one_rate=1,
        mean_absolute_error=0,
        blocking_eligible=True,
        reason="agreement thresholds met",
    )

    verdict = evaluate_regression(
        comparison,
        rules=[RegressionRule(metric="judge_correctness", minimum_delta=0)],
        judge_calibration=calibration,
    )

    assert not verdict.passed
    assert verdict.skipped_metrics == []
