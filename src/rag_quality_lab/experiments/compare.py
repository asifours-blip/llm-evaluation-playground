"""Deterministic baseline comparisons and regression gates."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, Field, model_validator

from rag_quality_lab.domain.models import CaseResult, ExperimentRecord
from rag_quality_lab.metrics.calibration import CalibrationResult

CASE_HIGHER_IS_BETTER = {"answer_f1", "retrieval_recall_at_k"}
CASE_LOWER_IS_BETTER = {"false_answer", "over_abstention"}


class MetricDelta(BaseModel):
    """Candidate minus baseline, with a baseline-relative percentage."""

    baseline: float
    candidate: float
    absolute: float
    percentage: float | None


class ComparisonResult(BaseModel):
    """Run-level deltas plus stable case-level changes."""

    baseline_id: str
    candidate_id: str
    metric_deltas: dict[str, MetricDelta]
    regressed_case_ids: list[str] = Field(default_factory=list)
    improved_case_ids: list[str] = Field(default_factory=list)


class RegressionRule(BaseModel):
    """Allowed candidate-minus-baseline interval for one metric."""

    metric: str
    minimum_delta: float | None = None
    maximum_delta: float | None = None

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if self.minimum_delta is None and self.maximum_delta is None:
            raise ValueError("regression rule requires a minimum or maximum delta")
        return self


class RegressionVerdict(BaseModel):
    """Gate decision with explicit failures and skipped judge metrics."""

    passed: bool
    failed_metrics: list[str] = Field(default_factory=list)
    skipped_metrics: list[str] = Field(default_factory=list)


class RegressionConfig(BaseModel):
    """Typed YAML wrapper for deterministic gate rules."""

    rules: list[RegressionRule]


class RegressionFixture(BaseModel):
    """Portable baseline/candidate pair evaluated by the real regression gate."""

    baseline: ExperimentRecord
    candidate: ExperimentRecord
    rules: list[RegressionRule]


def compare_experiments(
    baseline: ExperimentRecord, candidate: ExperimentRecord
) -> ComparisonResult:
    """Compare common summary metrics and deterministic per-case outcomes."""

    common_metrics = sorted(set(baseline.summary) & set(candidate.summary))
    deltas = {
        metric: _metric_delta(baseline.summary[metric], candidate.summary[metric])
        for metric in common_metrics
    }
    baseline_cases = _case_metric_means(baseline.case_results)
    candidate_cases = _case_metric_means(candidate.case_results)
    regressed: list[str] = []
    improved: list[str] = []
    for case_id in sorted(baseline_cases):
        if case_id not in candidate_cases:
            regressed.append(case_id)
            continue
        direction = _case_direction(baseline_cases[case_id], candidate_cases[case_id])
        if direction < 0:
            regressed.append(case_id)
        elif direction > 0:
            improved.append(case_id)
    return ComparisonResult(
        baseline_id=baseline.id,
        candidate_id=candidate.id,
        metric_deltas=deltas,
        regressed_case_ids=regressed,
        improved_case_ids=improved,
    )


def evaluate_regression(
    comparison: ComparisonResult,
    *,
    rules: Sequence[RegressionRule],
    judge_calibration: CalibrationResult | None = None,
) -> RegressionVerdict:
    """Evaluate configured gates, skipping uncalibrated judge metrics."""

    failed: list[str] = []
    skipped: list[str] = []
    for rule in rules:
        if rule.metric.startswith("judge_") and (
            judge_calibration is None or not judge_calibration.blocking_eligible
        ):
            skipped.append(rule.metric)
            continue
        delta = comparison.metric_deltas.get(rule.metric)
        if delta is None:
            failed.append(rule.metric)
            continue
        if rule.minimum_delta is not None and delta.absolute < rule.minimum_delta:
            failed.append(rule.metric)
            continue
        if rule.maximum_delta is not None and delta.absolute > rule.maximum_delta:
            failed.append(rule.metric)
    return RegressionVerdict(
        passed=not failed,
        failed_metrics=failed,
        skipped_metrics=skipped,
    )


def _metric_delta(baseline: float, candidate: float) -> MetricDelta:
    absolute = round(candidate - baseline, 12)
    percentage = None
    if baseline != 0:
        percentage = round(absolute / abs(baseline) * 100, 12)
    return MetricDelta(
        baseline=baseline,
        candidate=candidate,
        absolute=absolute,
        percentage=percentage,
    )


def _case_metric_means(results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        if result.status != "completed":
            continue
        for metric in CASE_HIGHER_IS_BETTER | CASE_LOWER_IS_BETTER:
            if metric in result.metrics:
                values[result.case_id][metric].append(result.metrics[metric])
    return {
        case_id: {
            metric: sum(metric_values) / len(metric_values)
            for metric, metric_values in metrics.items()
        }
        for case_id, metrics in values.items()
    }


def _case_direction(baseline: dict[str, float], candidate: dict[str, float]) -> int:
    regressed = any(
        candidate.get(metric, float("-inf")) < baseline_value
        for metric, baseline_value in baseline.items()
        if metric in CASE_HIGHER_IS_BETTER
    ) or any(
        candidate.get(metric, float("inf")) > baseline_value
        for metric, baseline_value in baseline.items()
        if metric in CASE_LOWER_IS_BETTER
    )
    improved = any(
        candidate.get(metric, float("-inf")) > baseline_value
        for metric, baseline_value in baseline.items()
        if metric in CASE_HIGHER_IS_BETTER
    ) or any(
        candidate.get(metric, float("inf")) < baseline_value
        for metric, baseline_value in baseline.items()
        if metric in CASE_LOWER_IS_BETTER
    )
    if regressed:
        return -1
    return int(improved)
