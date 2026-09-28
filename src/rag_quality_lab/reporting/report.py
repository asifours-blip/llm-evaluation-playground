"""Canonical JSON and self-contained HTML experiment reports."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, select_autoescape

from rag_quality_lab.domain.models import (
    UNLABELED,
    Answerability,
    CaseResult,
    ExperimentRecord,
    ExperimentStatus,
    canonical_hash,
)
from rag_quality_lab.experiments.compare import (
    SPLIT_ORDER,
    ComparisonResult,
    PairedComparison,
)
from rag_quality_lab.metrics.calibration import CalibrationResult

ReportBadge = Literal["mock", "pilot", "final"]
ANSWERABLE_ONLY_PREFIXES = ("retrieval_", "answer_", "over_abstention")
UNANSWERABLE_ONLY_METRICS = frozenset({"false_answer"})

# Standing caveats from the frozen v1.1 dataset review (docs/dataset-v1.1-review.md)
# and the archived strict-judge evidence run (docs/artifacts/final-evidence-summary-*.json).
# These carry forward into every report so a reader never mistakes a small offline
# or pilot run for a large, pristine, human-graded evaluation. Do not remove or
# soften any bullet without re-checking the archived artifacts it summarizes.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "Sample size is small: the frozen dataset has 48 cases total (32 dev, 16 "
    "frozen holdout); point estimates and their splits by category or label "
    "carry wide uncertainty.",
    "The most recent live strict-judge evidence run recorded a judge pass rate "
    "of roughly 61.5% (judge_pass_rate ~0.6146) and a false-answer rate of "
    "roughly 54.2%; treat those figures, not this run's numbers alone, as the "
    "reference point for how strict the judge is.",
    "The v1.1 holdout was frozen for reproducible scoring going forward, but it "
    "was not held out from every prior run of this project: cases now in the "
    "holdout split were evaluated before the freeze. Do not describe it as "
    "never having been seen.",
    "The dataset review in docs/dataset-v1.1-review.md was performed by an AI "
    "model at the repository owner's request, not by an independent human "
    "annotator.",
)


@dataclass(frozen=True)
class ReportPaths:
    """Files produced for one experiment report."""

    json: Path
    html: Path
    json_sha256: str
    html_sha256: str


def generate_reports(
    experiment: ExperimentRecord,
    output_dir: str | Path,
    *,
    badge: ReportBadge | None = None,
    comparison: ComparisonResult | None = None,
    calibration: CalibrationResult | None = None,
) -> ReportPaths:
    """Write canonical JSON and a self-contained HTML view."""

    report_badge = badge or ("mock" if experiment.identity.mode == "mock" else "pilot")
    if report_badge == "final":
        _validate_final_evidence(experiment, calibration)
    payload = _report_payload(
        experiment,
        badge=report_badge,
        comparison=comparison,
        calibration=calibration,
    )
    return _write_report(payload, Path(output_dir), experiment.id, "report.html.jinja2")


def generate_comparison_report(
    comparison: PairedComparison, output_dir: str | Path
) -> ReportPaths:
    """Write a paired arm comparison as canonical JSON and self-contained HTML."""

    payload = comparison.model_dump(mode="json")
    name = "comparison-" + canonical_hash(
        [
            comparison.baseline_id,
            comparison.baseline_config_id,
            comparison.candidate_id,
            comparison.candidate_config_id,
        ]
    )[:16]
    return _write_report(payload, Path(output_dir), name, "comparison.html.jinja2")


def _write_report(
    payload: dict[str, Any], destination: Path, name: str, template: str
) -> ReportPaths:
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{name}.json"
    html_path = destination / f"{name}.html"
    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    json_path.write_text(json_text, encoding="utf-8", newline="\n")

    template_dir = Path(__file__).parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(enabled_extensions=("html", "jinja2")),
    )
    rendered_html = environment.get_template(template).render(report=payload)
    html_text = "\n".join(line.rstrip() for line in rendered_html.splitlines()) + "\n"
    html_path.write_text(html_text, encoding="utf-8", newline="\n")
    return ReportPaths(
        json=json_path,
        html=html_path,
        json_sha256=_sha256(json_text),
        html_sha256=_sha256(html_text),
    )


def _validate_final_evidence(
    experiment: ExperimentRecord, calibration: CalibrationResult | None
) -> None:
    if experiment.identity.mode != "live":
        raise ValueError("final evidence requires a live experiment")
    if experiment.status is not ExperimentStatus.COMPLETED:
        raise ValueError("final evidence requires a completed experiment")
    if experiment.identity.dirty:
        raise ValueError("final evidence requires a clean git identity")
    if any(result.status != "completed" for result in experiment.case_results):
        raise ValueError("final evidence cannot contain failed cases")
    if any(result.http_request_count is None for result in experiment.case_results) or any(
        call.http_request_count is None for call in experiment.embedding_calls
    ):
        raise ValueError("final evidence requires complete HTTP request counts")
    if calibration is None or not calibration.blocking_eligible:
        raise ValueError("final evidence requires eligible human judge calibration")


def _report_payload(
    experiment: ExperimentRecord,
    *,
    badge: ReportBadge,
    comparison: ComparisonResult | None,
    calibration: CalibrationResult | None,
) -> dict[str, Any]:
    results = experiment.case_results
    failures = [
        result.model_dump(mode="json")
        for result in results
        if result.status not in {"completed", "cancelled"}
    ]
    payload: dict[str, Any] = {
        "id": experiment.id,
        "status": experiment.status.value,
        "badge": badge,
        "identity": experiment.identity.model_dump(mode="json"),
        "summary": experiment.summary,
        "system": _system_metrics(results),
        "category_breakdown": _category_breakdown(results),
        "split_breakdown": _split_breakdown(experiment),
        "dataset_labels": _dataset_labels(results),
        "failures": failures,
        "case_results": [result.model_dump(mode="json") for result in results],
        "limitations": list(KNOWN_LIMITATIONS),
        "baseline_comparison": (
            comparison.model_dump(mode="json") if comparison is not None else None
        ),
        "judge_calibration": (
            calibration.model_dump(mode="json") if calibration is not None else None
        ),
    }
    if experiment.embedding_calls:
        # Additive key: reports without remote embedding calls keep their shape.
        payload["embedding_calls"] = [
            call.model_dump(mode="json") for call in experiment.embedding_calls
        ]
    if len(experiment.attempts) > 1:
        # Additive keys: a resumed experiment lists every attempt's commit and
        # warns when results were produced by different code versions.
        payload["run_attempts"] = [
            attempt.model_dump(mode="json") for attempt in experiment.attempts
        ]
    if comparison is not None and comparison.warning is not None:
        # Additive key: shown first because the deltas cannot be attributed.
        payload["comparison_warning"] = comparison.warning
    warning = experiment.code_version_warning()
    if warning is not None:
        payload["code_version_warning"] = warning
    cancelled = [result for result in results if result.status == "cancelled"]
    if cancelled:
        # Additive key: case arms stopped by cancellation are counted apart
        # from failures and never scored.
        payload["cancelled_cases"] = [result.model_dump(mode="json") for result in cancelled]
    if experiment.unknown_calls:
        # Additive key: calls sent before an interruption but never settled,
        # charged at their reserved cap.
        payload["unknown_calls"] = [
            call.model_dump(mode="json") for call in experiment.unknown_calls
        ]
    return payload


def _system_metrics(
    results: Sequence[CaseResult],
) -> dict[str, float | int | bool | None]:
    latencies = [result.latency_ms for result in results if result.status == "completed"]
    usages = [
        usage
        for result in results
        for usage in (result.usage, result.judge_usage)
        if usage is not None
    ]
    request_counts = [result.http_request_count for result in results]
    request_count_complete = all(count is not None for count in request_counts)
    return {
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": _nearest_rank(latencies, 0.95),
        "input_tokens": sum(usage.input_tokens for usage in usages),
        "output_tokens": sum(usage.output_tokens for usage in usages),
        "total_tokens": sum(usage.total_tokens for usage in usages),
        "total_cost": float(sum(result.cost for result in results)),
        "failure_count": sum(
            result.status not in {"completed", "cancelled"} for result in results
        ),
        "http_request_count": (
            sum(count for count in request_counts if count is not None)
            if request_count_complete
            else None
        ),
        "http_request_count_complete": request_count_complete,
    }


def _category_breakdown(results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        if result.status != "completed":
            continue
        category = result.category or "uncategorized"
        for metric_name, value in result.metrics.items():
            metrics[category][metric_name].append(value)
    return {
        category: {
            metric_name: statistics.fmean(values)
            for metric_name, values in sorted(category_metrics.items())
        }
        for category, category_metrics in sorted(metrics.items())
    }


def _split_breakdown(experiment: ExperimentRecord) -> dict[str, dict[str, Any]]:
    """Per-split, per-configuration metric means; splits are never pooled.

    Retrieval, answer, and over-abstention metrics average answerable cases
    and false answers average unanswerable cases, matching the run summary.
    Only a holdout whose frozen hash was verified at run time is held-out
    evidence; dev and unlabeled cases never are.
    """

    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for result in experiment.case_results:
        groups[result.split or UNLABELED].append(result)
    freeze_hash = experiment.identity.holdout_freeze_hash
    breakdown: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        results = groups.get(split)
        if not results:
            continue
        entry: dict[str, Any] = {
            "case_count": len({result.case_id for result in results}),
            "result_count": len(results),
            "metrics": _config_metric_means(results),
        }
        if split == "holdout" and freeze_hash is not None:
            entry.update(
                held_out=True,
                holdout_freeze_hash=freeze_hash,
                note=f"frozen holdout (hash {freeze_hash}); held-out evidence",
            )
        elif split == "holdout":
            entry.update(
                held_out=False,
                note="holdout split was not frozen when this ran; not held-out evidence",
            )
        elif split == "dev":
            entry.update(
                held_out=False,
                note="dev split used for tuning; not held-out evidence",
            )
        else:
            entry.update(
                held_out=False,
                note=f"split {UNLABELED} (unlabeled); not held-out evidence",
            )
        breakdown[split] = entry
    return breakdown


def _config_metric_means(results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        if result.status != "completed":
            continue
        answerable = result.answerability is not Answerability.UNANSWERABLE
        for metric, value in result.metrics.items():
            if metric.startswith(ANSWERABLE_ONLY_PREFIXES) and not answerable:
                continue
            if metric in UNANSWERABLE_ONLY_METRICS and answerable:
                continue
            values[result.config_id][metric].append(value)
    return {
        config_id: {
            metric: statistics.fmean(metric_values)
            for metric, metric_values in sorted(metrics.items())
        }
        for config_id, metrics in sorted(values.items())
    }


def _dataset_labels(results: Sequence[CaseResult]) -> dict[str, dict[str, int]]:
    """Count distinct cases per label value; absent labels count as unlabeled."""

    cases: dict[str, CaseResult] = {}
    for result in results:
        cases.setdefault(result.case_id, result)
    labels: dict[str, dict[str, int]] = {
        "split": {},
        "difficulty": {},
        "review_status": {},
        "answerability": {},
    }
    for result in cases.values():
        for key, value in (
            ("split", result.split),
            ("difficulty", result.difficulty),
            ("review_status", result.review_status),
            (
                "answerability",
                result.answerability.value if result.answerability is not None else None,
            ),
        ):
            label = value or UNLABELED
            labels[key][label] = labels[key].get(label, 0) + 1
    return labels


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
