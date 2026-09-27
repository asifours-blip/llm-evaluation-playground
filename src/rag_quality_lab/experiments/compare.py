"""Deterministic baseline comparisons, paired arm comparisons, and regression gates."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from rag_quality_lab.domain.models import (
    UNLABELED,
    CaseResult,
    ExperimentConfig,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
    ParameterDifference,
    RetrievalConfig,
)
from rag_quality_lab.metrics.calibration import CalibrationResult

CASE_HIGHER_IS_BETTER = {"answer_f1", "retrieval_recall_at_k"}
CASE_LOWER_IS_BETTER = {"false_answer", "over_abstention"}
CONFOUNDED_WARNING = "本次比较混杂多个变量，差异不能归因于单一改动"
# Fields that locate inputs or tune execution but do not change what is
# measured; input content is compared through the dataset and corpus hashes.
NON_PARAMETER_FIELDS = frozenset(
    {
        "name",
        "dataset_path",
        "knowledge_base_path",
        "database_path",
        "artifact_dir",
        "pricing_path",
        "max_workers",
        "random_seed",
    }
)
NON_PARAMETER_PROVIDER_FIELDS = frozenset({"api_key_env"})
SPLIT_ORDER = ("dev", "holdout", UNLABELED)


class ComparisonRefused(ValueError):
    """Two configurations were not evaluated under identical conditions."""


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
    config_differences: list[ParameterDifference] = Field(default_factory=list)
    warning: str | None = None


class RankedEvidence(BaseModel):
    """One retrieved chunk at its 1-based rank."""

    rank: int = Field(ge=1)
    chunk_id: str
    document_id: str
    score: float
    text: str


class RankChange(BaseModel):
    """A chunk retrieved by both configurations at different ranks."""

    chunk_id: str
    baseline_rank: int
    candidate_rank: int


class PairedCase(BaseModel):
    """Side-by-side retrieval evidence and metric deltas for one question."""

    case_id: str
    question: str
    split: str
    model: str
    baseline_status: str
    candidate_status: str
    baseline_hits: list[RankedEvidence]
    candidate_hits: list[RankedEvidence]
    shared_chunk_ids: list[str]
    baseline_only_chunk_ids: list[str]
    candidate_only_chunk_ids: list[str]
    rank_changes: list[RankChange]
    metric_deltas: dict[str, MetricDelta]
    direction: Literal["improved", "regressed", "unchanged", "incomparable"]


class PairedComparison(BaseModel):
    """Per-question comparison of two configurations under identical conditions."""

    baseline_id: str
    candidate_id: str
    baseline_config_id: str
    candidate_config_id: str
    conditions: dict[str, str]
    config_differences: list[ParameterDifference]
    warning: str | None = None
    split_deltas: dict[str, dict[str, MetricDelta]]
    cases: list[PairedCase]


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
    """Manifest for rerunning a real offline pipeline against committed evidence."""

    config_path: Path
    baseline_report_path: Path
    rules: list[RegressionRule]


def compare_experiments(
    baseline: ExperimentRecord, candidate: ExperimentRecord
) -> ComparisonResult:
    """Compare common summary metrics and deterministic per-case outcomes."""

    if baseline.status is not ExperimentStatus.COMPLETED or (
        candidate.status is not ExperimentStatus.COMPLETED
    ):
        raise ValueError("comparisons require two completed experiments")
    if baseline.identity.dataset_hash != candidate.identity.dataset_hash:
        raise ValueError("comparisons require matching dataset hashes")
    differences = parameter_differences(baseline.identity, candidate.identity)
    common_metrics = sorted(set(baseline.summary) & set(candidate.summary))
    deltas = {
        metric: _metric_delta(baseline.summary[metric], candidate.summary[metric])
        for metric in common_metrics
    }
    baseline_cases = _case_metric_means(baseline.case_results)
    candidate_cases = _case_metric_means(candidate.case_results)
    regressed: list[str] = []
    improved: list[str] = []
    for case_key in sorted(baseline_cases):
        display_key = "::".join(case_key)
        if case_key not in candidate_cases:
            regressed.append(display_key)
            continue
        direction = _case_direction(
            baseline_cases[case_key], candidate_cases[case_key]
        )
        if direction < 0:
            regressed.append(display_key)
        elif direction > 0:
            improved.append(display_key)
    return ComparisonResult(
        baseline_id=baseline.id,
        candidate_id=candidate.id,
        metric_deltas=deltas,
        regressed_case_ids=regressed,
        improved_case_ids=improved,
        config_differences=differences,
        warning=confounding_warning(differences),
    )


def confounding_warning(differences: Sequence[ParameterDifference]) -> str | None:
    """Warn when more than one parameter differs between compared configurations."""

    return CONFOUNDED_WARNING if len(differences) > 1 else None


def pairing_refusals(
    baseline: ExperimentRecord,
    candidate: ExperimentRecord,
    baseline_config_id: str,
    candidate_config_id: str,
) -> list[str]:
    """Name every condition that differs between two configuration arms.

    A pair is comparable only with the same knowledge-base corpus, dataset
    version and content, question set, and random seed.
    """

    left = baseline.identity
    right = candidate.identity
    reasons: list[str] = []
    if left.corpus_hash is None or right.corpus_hash is None:
        missing = [
            record.id
            for record in (baseline, candidate)
            if record.identity.corpus_hash is None
        ]
        reasons.append(
            "knowledge-base corpus hash is not recorded for " + ", ".join(missing)
        )
    elif left.corpus_hash != right.corpus_hash:
        reasons.append(
            f"knowledge-base corpus differs ({_short(left.corpus_hash)} vs "
            f"{_short(right.corpus_hash)})"
        )
    versions_known = left.dataset_version is not None and right.dataset_version is not None
    if versions_known and left.dataset_version != right.dataset_version:
        reasons.append(
            f"dataset version differs ({left.dataset_version} vs {right.dataset_version})"
        )
    elif left.dataset_hash != right.dataset_hash:
        reasons.append(
            f"dataset content differs (dataset_hash {_short(left.dataset_hash)} vs "
            f"{_short(right.dataset_hash)})"
        )
    left_questions = _arm_questions(baseline, baseline_config_id)
    right_questions = _arm_questions(candidate, candidate_config_id)
    for record, config_id, questions in (
        (baseline, baseline_config_id, left_questions),
        (candidate, candidate_config_id, right_questions),
    ):
        if not questions:
            reasons.append(f"{record.id} has no results for configuration {config_id}")
    only_left = sorted(set(left_questions) - set(right_questions))
    only_right = sorted(set(right_questions) - set(left_questions))
    if only_left or only_right:
        reasons.append(
            "question set differs (only in baseline: "
            f"{', '.join(only_left) or 'none'}; only in candidate: "
            f"{', '.join(only_right) or 'none'})"
        )
    changed = sorted(
        case_id
        for case_id in set(left_questions) & set(right_questions)
        if left_questions[case_id] != right_questions[case_id]
    )
    if changed:
        reasons.append("question set differs (question text: " + ", ".join(changed) + ")")
    if left.random_seed != right.random_seed:
        reasons.append(f"random seed differs ({left.random_seed} vs {right.random_seed})")
    return reasons


def require_pairable(
    baseline: ExperimentRecord,
    candidate: ExperimentRecord,
    baseline_config_id: str,
    candidate_config_id: str,
) -> None:
    """Refuse a paired comparison unless every controlled condition matches."""

    reasons = pairing_refusals(baseline, candidate, baseline_config_id, candidate_config_id)
    if reasons:
        raise ComparisonRefused(
            f"cannot pair {baseline.id}:{baseline_config_id} with "
            f"{candidate.id}:{candidate_config_id}: " + "; ".join(reasons)
        )


def parameter_differences(
    baseline: ExperimentIdentity,
    candidate: ExperimentIdentity,
    baseline_config_id: str | None = None,
    candidate_config_id: str | None = None,
) -> list[ParameterDifference]:
    """List every parameter whose value differs, sorted by path.

    With config IDs, only the two selected retrieval arms are compared (as
    ``retrieval.*``); otherwise the complete retrieval matrices are. Code
    version, prompt templates, and random seed count as parameters too.
    """

    left = parameter_config(baseline.config)
    right = parameter_config(candidate.config)
    prompt_variants: set[str] | None = None
    if baseline_config_id is not None and candidate_config_id is not None:
        left_arm = _arm_config(baseline, baseline_config_id)
        right_arm = _arm_config(candidate, candidate_config_id)
        left["retrieval"] = left_arm
        right["retrieval"] = right_arm
        prompt_variants = {
            str(arm.get("prompt_variant")) for arm in (left_arm, right_arm)
        }
    differences = [
        ParameterDifference(
            path=path,
            baseline=_value_at(left, path),
            candidate=_value_at(right, path),
        )
        for path in diff_paths(left, right)
    ]
    if baseline.commit_sha != candidate.commit_sha:
        differences.append(
            ParameterDifference(
                path="code_version.commit_sha",
                baseline=baseline.commit_sha,
                candidate=candidate.commit_sha,
            )
        )
    left_prompts = _selected(baseline.prompt_hashes, prompt_variants)
    right_prompts = _selected(candidate.prompt_hashes, prompt_variants)
    differences.extend(
        ParameterDifference(
            path=f"prompts.{path}",
            baseline=_value_at(left_prompts, path),
            candidate=_value_at(right_prompts, path),
        )
        for path in diff_paths(left_prompts, right_prompts)
    )
    if baseline.prompt_version != candidate.prompt_version:
        differences.append(
            ParameterDifference(
                path="prompts.version",
                baseline=baseline.prompt_version,
                candidate=candidate.prompt_version,
            )
        )
    if baseline.random_seed != candidate.random_seed:
        differences.append(
            ParameterDifference(
                path="random_seed",
                baseline=baseline.random_seed,
                candidate=candidate.random_seed,
            )
        )
    return sorted(differences, key=lambda difference: difference.path)


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    """Re-validate a stored config so fields added later take their defaults."""

    try:
        return ExperimentConfig.model_validate(config).model_dump(mode="json")
    except ValidationError:
        return dict(config)


def parameter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Canonical config without fields that do not change what is measured."""

    payload = {
        key: value
        for key, value in canonical_config(config).items()
        if key not in NON_PARAMETER_FIELDS
    }
    provider = payload.get("provider")
    if isinstance(provider, dict):
        payload["provider"] = {
            key: value
            for key, value in provider.items()
            if key not in NON_PARAMETER_PROVIDER_FIELDS
        }
    return payload


def diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    """Dotted paths at which two JSON-like values differ."""

    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(diff_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            path
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True))
            for path in diff_paths(left_item, right_item, f"{prefix}[{index}]")
        ]
    return [] if left == right else [prefix]


def compare_arms(
    baseline: ExperimentRecord,
    candidate: ExperimentRecord,
    baseline_config_id: str,
    candidate_config_id: str,
) -> PairedComparison:
    """Compare two configuration arms question by question.

    Refuses unless both arms share corpus, dataset version and content,
    question set, and random seed. Every other differing parameter is listed;
    more than one yields :data:`CONFOUNDED_WARNING`. Deltas are reported per
    dataset split, so dev results are never pooled with holdout results.
    """

    require_pairable(baseline, candidate, baseline_config_id, candidate_config_id)
    differences = parameter_differences(
        baseline.identity, candidate.identity, baseline_config_id, candidate_config_id
    )
    left_results = _arm_results(baseline, baseline_config_id)
    right_results = _arm_results(candidate, candidate_config_id)
    cases = [
        _paired_case(left_results[key], right_results[key])
        for key in sorted(set(left_results) & set(right_results))
    ]
    return PairedComparison(
        baseline_id=baseline.id,
        candidate_id=candidate.id,
        baseline_config_id=baseline_config_id,
        candidate_config_id=candidate_config_id,
        conditions={
            "corpus_hash": baseline.identity.corpus_hash or "",
            "dataset_version": baseline.identity.dataset_version or UNLABELED,
            "dataset_hash": baseline.identity.dataset_hash,
            "random_seed": str(baseline.identity.random_seed),
            "question_count": str(len({case.case_id for case in cases})),
        },
        config_differences=differences,
        warning=confounding_warning(differences),
        split_deltas=_split_deltas(left_results, right_results),
        cases=cases,
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


def _short(value: str) -> str:
    return value[:12]


def _arm_questions(experiment: ExperimentRecord, config_id: str) -> dict[str, str]:
    return {
        result.case_id: result.question
        for result in experiment.case_results
        if result.config_id == config_id
    }


def _arm_results(
    experiment: ExperimentRecord, config_id: str
) -> dict[tuple[str, str], CaseResult]:
    return {
        (result.case_id, result.model): result
        for result in experiment.case_results
        if result.config_id == config_id
    }


def _arm_config(identity: ExperimentIdentity, config_id: str) -> dict[str, Any]:
    arms = canonical_config(identity.config).get("retrieval", [])
    for arm in arms if isinstance(arms, list) else []:
        try:
            if RetrievalConfig.model_validate(arm).config_id == config_id:
                return dict(arm)
        except ValidationError:
            continue
    raise ValueError(f"configuration {config_id} is not in experiment {identity.name}")


def _selected(hashes: dict[str, str], variants: set[str] | None) -> dict[str, str]:
    if variants is None:
        return dict(hashes)
    return {key: value for key, value in hashes.items() if key in variants}


def _value_at(payload: Any, path: str) -> Any:
    value = payload
    for part in path.replace("[", ".[").split("."):
        if not part:
            continue
        if part.startswith("[") and isinstance(value, list):
            index = int(part[1:-1])
            value = value[index] if index < len(value) else None
        elif isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _ranked(result: CaseResult) -> list[RankedEvidence]:
    return [
        RankedEvidence(
            rank=rank,
            chunk_id=hit.chunk.id,
            document_id=hit.chunk.document_id,
            score=hit.score,
            text=hit.chunk.text,
        )
        for rank, hit in enumerate(result.retrieval_hits, start=1)
    ]


def _paired_case(left: CaseResult, right: CaseResult) -> PairedCase:
    left_hits = _ranked(left)
    right_hits = _ranked(right)
    left_ranks = {hit.chunk_id: hit.rank for hit in left_hits}
    right_ranks = {hit.chunk_id: hit.rank for hit in right_hits}
    shared = [hit.chunk_id for hit in left_hits if hit.chunk_id in right_ranks]
    comparable = left.status == "completed" and right.status == "completed"
    deltas: dict[str, MetricDelta] = {}
    direction: Literal["improved", "regressed", "unchanged", "incomparable"] = (
        "incomparable"
    )
    if comparable:
        deltas = {
            metric: _metric_delta(left.metrics[metric], right.metrics[metric])
            for metric in sorted(set(left.metrics) & set(right.metrics))
        }
        tracked = CASE_HIGHER_IS_BETTER | CASE_LOWER_IS_BETTER
        outcome = _case_direction(
            {key: value for key, value in left.metrics.items() if key in tracked},
            {key: value for key, value in right.metrics.items() if key in tracked},
        )
        direction = "regressed" if outcome < 0 else "improved" if outcome else "unchanged"
    return PairedCase(
        case_id=left.case_id,
        question=left.question,
        split=left.split or UNLABELED,
        model=left.model,
        baseline_status=left.status,
        candidate_status=right.status,
        baseline_hits=left_hits,
        candidate_hits=right_hits,
        shared_chunk_ids=shared,
        baseline_only_chunk_ids=[
            hit.chunk_id for hit in left_hits if hit.chunk_id not in right_ranks
        ],
        candidate_only_chunk_ids=[
            hit.chunk_id for hit in right_hits if hit.chunk_id not in left_ranks
        ],
        rank_changes=[
            RankChange(
                chunk_id=chunk_id,
                baseline_rank=left_ranks[chunk_id],
                candidate_rank=right_ranks[chunk_id],
            )
            for chunk_id in shared
            if left_ranks[chunk_id] != right_ranks[chunk_id]
        ],
        metric_deltas=deltas,
        direction=direction,
    )


def _split_deltas(
    left: dict[tuple[str, str], CaseResult],
    right: dict[tuple[str, str], CaseResult],
) -> dict[str, dict[str, MetricDelta]]:
    """Mean metric deltas per split over questions both arms completed."""

    grouped: dict[str, list[tuple[CaseResult, CaseResult]]] = {}
    for key in sorted(set(left) & set(right)):
        if left[key].status == "completed" and right[key].status == "completed":
            grouped.setdefault(left[key].split or UNLABELED, []).append(
                (left[key], right[key])
            )
    deltas: dict[str, dict[str, MetricDelta]] = {}
    for split in SPLIT_ORDER:
        pairs = grouped.get(split)
        if not pairs:
            continue
        metrics = sorted(
            set.intersection(
                *(set(pair[0].metrics) & set(pair[1].metrics) for pair in pairs)
            )
        )
        deltas[split] = {
            metric: _metric_delta(
                sum(pair[0].metrics[metric] for pair in pairs) / len(pairs),
                sum(pair[1].metrics[metric] for pair in pairs) / len(pairs),
            )
            for metric in metrics
        }
    return deltas


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


def _case_metric_means(
    results: Sequence[CaseResult],
) -> dict[tuple[str, str, str], dict[str, float]]:
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    for result in results:
        if result.status != "completed":
            continue
        key = (result.case_id, result.config_id, result.model)
        if key in values:
            raise ValueError(f"duplicate completed comparison key: {'::'.join(key)}")
        values[key] = {}
        for metric in CASE_HIGHER_IS_BETTER | CASE_LOWER_IS_BETTER:
            if metric in result.metrics:
                values[key][metric] = result.metrics[metric]
    return values


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
