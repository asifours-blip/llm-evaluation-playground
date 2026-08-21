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

from rag_quality_lab.domain.models import CaseResult, ExperimentRecord, ExperimentStatus
from rag_quality_lab.experiments.compare import ComparisonResult
from rag_quality_lab.metrics.calibration import CalibrationResult

ReportBadge = Literal["mock", "pilot", "final"]


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
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{experiment.id}.json"
    html_path = destination / f"{experiment.id}.html"
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
    rendered_html = environment.get_template("report.html.jinja2").render(report=payload)
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
    if any(result.http_request_count is None for result in experiment.case_results):
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
        result.model_dump(mode="json") for result in results if result.status != "completed"
    ]
    return {
        "id": experiment.id,
        "status": experiment.status.value,
        "badge": badge,
        "identity": experiment.identity.model_dump(mode="json"),
        "summary": experiment.summary,
        "system": _system_metrics(results),
        "category_breakdown": _category_breakdown(results),
        "failures": failures,
        "case_results": [result.model_dump(mode="json") for result in results],
        "baseline_comparison": (
            comparison.model_dump(mode="json") if comparison is not None else None
        ),
        "judge_calibration": (
            calibration.model_dump(mode="json") if calibration is not None else None
        ),
    }


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
        "failure_count": sum(result.status != "completed" for result in results),
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


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
