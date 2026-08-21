"""Coordinator for reproducible offline and live RAG experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from rag_quality_lab.config.loaders import load_yaml_model, validate_dataset_corpus
from rag_quality_lab.domain.models import (
    Answerability,
    CaseResult,
    Document,
    EvaluationCase,
    EvaluationDataset,
    ExperimentConfig,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
    PricingConfig,
    ProviderResponse,
    RetrievalConfig,
    RetrievalHit,
    StructuredAnswer,
    TokenUsage,
)
from rag_quality_lab.experiments.budget import (
    BudgetExceeded,
    BudgetLedger,
    PlannedCall,
    calculate_actual_cost,
    preflight_budget,
)
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.metrics.abstention import (
    AbstentionObservation,
    is_effective_abstention,
    summarize_abstention,
)
from rag_quality_lab.metrics.answer import (
    bilingual_f1,
    normalized_exact_match,
    semantic_similarity,
)
from rag_quality_lab.metrics.retrieval import (
    context_hit_rate,
    recall_at_k,
    reciprocal_rank,
)
from rag_quality_lab.prompts.engine import PromptEngine
from rag_quality_lab.providers.base import ChatProvider, EmbeddingProvider, JudgeProvider
from rag_quality_lab.providers.openai_compatible import (
    GENERATION_INPUT_TOKEN_CAP,
    GENERATION_OUTPUT_TOKEN_CAP,
    JUDGE_INPUT_TOKEN_CAP,
    JUDGE_OUTPUT_TOKEN_CAP,
    REPAIR_PROMPT_TOKEN_ALLOWANCE,
    ProviderError,
)
from rag_quality_lab.retrieval.index import InMemoryIndex, chunk_document, load_documents

FailurePhase = Literal["retrieval", "generation", "metrics", "judge"]


@dataclass(frozen=True)
class ProviderBundle:
    """Provider implementations used by one runner invocation."""

    embedding: EmbeddingProvider
    chat: ChatProvider
    judge: JudgeProvider | None = None


@dataclass(frozen=True)
class _Task:
    case: EvaluationCase
    retrieval: RetrievalConfig
    config_id: str
    index: InMemoryIndex
    instructions: str


@dataclass(frozen=True)
class _TaskOutput:
    result: CaseResult


class _TaskFailure(RuntimeError):
    """A case-local failure tagged with the pipeline phase that raised it."""

    def __init__(
        self,
        phase: FailurePhase,
        cause: Exception,
        *,
        response: ProviderResponse[StructuredAnswer] | None = None,
        hits: Sequence[RetrievalHit] = (),
    ) -> None:
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause
        self.response = response
        self.hits = list(hits)


def run_experiment(
    config: ExperimentConfig,
    providers: ProviderBundle,
    dataset: EvaluationDataset,
) -> ExperimentRecord:
    """Run all configured case arms and persist each completed outcome."""

    if config.provider.judge_model is not None and providers.judge is None:
        raise ValueError("judge_model requires a judge provider")
    documents = load_documents(config.knowledge_base_path)
    validate_dataset_corpus(dataset, documents)
    prompt_engine = PromptEngine()
    identity = _experiment_identity(config, dataset, prompt_engine)
    tasks = list(_tasks(config, dataset, documents, providers.embedding, prompt_engine))

    with ExperimentStore(config.database_path) as store:
        experiment_id = store.create_experiment(identity)
        pricing, ledger = _prepare_live_budget(config, tasks)
        if config.mode == "live" and ledger is None:
            summary = _summary([], dataset)
            store.finish_experiment(
                experiment_id,
                ExperimentStatus.BUDGET_EXCEEDED,
                summary=summary,
            )
            return store.get_experiment(experiment_id)

        status = ExperimentStatus.COMPLETED
        try:
            budget_stopped = _coordinate_tasks(
                tasks=iter(tasks),
                config=config,
                providers=providers,
                store=store,
                experiment_id=experiment_id,
                ledger=ledger,
                pricing=pricing,
            )
            if budget_stopped:
                status = ExperimentStatus.BUDGET_EXCEEDED
            running_record = store.get_experiment(experiment_id)
            summary = _summary(running_record.case_results, dataset)
            store.finish_experiment(experiment_id, status, summary=summary)
        except Exception:
            store.finish_experiment(experiment_id, ExperimentStatus.FAILED)
            raise
        return store.get_experiment(experiment_id)


def _prepare_live_budget(
    config: ExperimentConfig, tasks: Sequence[_Task]
) -> tuple[PricingConfig | None, BudgetLedger | None]:
    if config.mode == "mock":
        return None, None
    if config.pricing_path is None:
        raise ValueError("live experiments require a pricing file")
    pricing = load_yaml_model(config.pricing_path, PricingConfig)
    decision = preflight_budget(
        planned=planned_calls(config, len(tasks)),
        pricing=pricing,
        budget=config.budget,
    )
    if not decision.allowed:
        return pricing, None
    return pricing, BudgetLedger(budget=config.budget, pricing=pricing)


def _coordinate_tasks(
    *,
    tasks: Iterator[_Task],
    config: ExperimentConfig,
    providers: ProviderBundle,
    store: ExperimentStore,
    experiment_id: str,
    ledger: BudgetLedger | None,
    pricing: PricingConfig | None,
) -> bool:
    pending: dict[Future[_TaskOutput], tuple[_Task, list[Decimal]]] = {}
    no_more_tasks = False
    budget_stopped = False
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        while pending or not no_more_tasks:
            while (
                len(pending) < config.max_workers
                and not no_more_tasks
                and not budget_stopped
            ):
                try:
                    task = next(tasks)
                except StopIteration:
                    no_more_tasks = True
                    break
                reservations: list[Decimal] = []
                if ledger is not None:
                    try:
                        reservations = ledger.reserve_many(_case_planned_calls(config))
                    except BudgetExceeded:
                        budget_stopped = True
                        break
                future = executor.submit(_evaluate_task, task, config, providers)
                pending[future] = (task, reservations)

            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(
                completed,
                key=lambda item: (
                    pending[item][0].case.id,
                    pending[item][0].config_id,
                ),
            ):
                task, reservations = pending.pop(future)
                try:
                    output = future.result()
                    result = output.result
                except _TaskFailure as failure:
                    actual_cost, estimated_cost = _reconcile_failed_reservations(
                        failure,
                        reservations,
                        config,
                        ledger,
                    )
                    result = _failed_case_result(
                        task,
                        config,
                        failure,
                        cost=actual_cost + estimated_cost,
                        cost_estimated=bool(estimated_cost),
                    )
                    store.record_case_result(experiment_id, result)
                    continue
                if ledger is not None and pricing is not None:
                    call_usages = _result_call_usages(result, config)
                    try:
                        actual_cost = ledger.settle_many(reservations, call_usages)
                    except BudgetExceeded:
                        actual_cost = sum(
                            (
                                calculate_actual_cost(usage, pricing.models[model])
                                for model, usage in call_usages
                            ),
                            start=Decimal("0"),
                        )
                        budget_stopped = True
                    result = result.model_copy(update={"cost": actual_cost})
                store.record_case_result(experiment_id, result)
    return budget_stopped


def _tasks(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    documents: Sequence[Document],
    embedding_provider: EmbeddingProvider,
    prompt_engine: PromptEngine,
) -> Iterator[_Task]:
    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document.id,
                document.text,
                chunk_size=retrieval.chunk_size,
                chunk_overlap=retrieval.chunk_overlap,
            )
        ]
        index = InMemoryIndex.from_chunks(
            chunks,
            embedding_provider,
            model=config.provider.embedding_model,
            cache_path=config.artifact_dir / f"{config_id}.embeddings.json",
        )
        for case in dataset.cases:
            yield _Task(
                case=case,
                retrieval=retrieval,
                config_id=config_id,
                index=index,
                instructions=prompt_engine.instructions(retrieval.prompt_variant),
            )


def _evaluate_task(
    task: _Task, config: ExperimentConfig, providers: ProviderBundle
) -> _TaskOutput:
    try:
        hits = task.index.search(task.case.question, top_k=task.retrieval.top_k)
    except Exception as error:
        raise _TaskFailure("retrieval", error) from error
    contexts = [f"[{hit.chunk.id}]\n{hit.chunk.text}" for hit in hits]
    try:
        response = providers.chat.answer(
            task.case.question,
            contexts,
            model=config.provider.chat_model,
            instructions=task.instructions,
        )
    except Exception as error:
        raise _TaskFailure("generation", error, hits=hits) from error
    try:
        answer_vectors = providers.embedding.embed(
            [response.parsed.answer, task.case.reference_answer],
            model=config.provider.embedding_model,
        )
    except Exception as error:
        raise _TaskFailure("metrics", error, response=response, hits=hits) from error
    expected_answerable = task.case.answerability is Answerability.ANSWERABLE
    effective_abstention = is_effective_abstention(
        abstained=response.parsed.abstained,
        answer=response.parsed.answer,
    )
    abstention_correct = effective_abstention != expected_answerable
    metrics = {
        "retrieval_recall_at_k": recall_at_k(
            hits,
            task.case.expected_document_ids,
            k=task.retrieval.top_k,
        ),
        "retrieval_mrr": reciprocal_rank(hits, task.case.expected_document_ids),
        "retrieval_context_hit_rate": context_hit_rate(
            hits, task.case.reference_evidence
        ),
        "answer_exact_match": normalized_exact_match(
            response.parsed.answer, task.case.reference_answer
        ),
        "answer_f1": bilingual_f1(
            response.parsed.answer, task.case.reference_answer
        ),
        "answer_semantic_similarity": semantic_similarity(
            answer_vectors[0], answer_vectors[1]
        ),
        "abstention_correct": float(abstention_correct),
        "false_answer": float(not expected_answerable and not effective_abstention),
        "over_abstention": float(expected_answerable and effective_abstention),
    }
    judge_response = None
    if config.provider.judge_model is not None and providers.judge is not None:
        try:
            judge_response = providers.judge.judge(
                task.case.question,
                task.case.reference_answer,
                response.parsed.answer,
                [hit.chunk.text for hit in hits],
                model=config.provider.judge_model,
            )
        except Exception as error:
            raise _TaskFailure("judge", error, response=response, hits=hits) from error
        metrics.update(
            {
                "judge_score": float(judge_response.parsed.score),
                "judge_pass": float(judge_response.parsed.passed),
            }
        )
    result = CaseResult(
        case_id=task.case.id,
        question=task.case.question,
        reference_answer=task.case.reference_answer,
        reference_evidence=task.case.reference_evidence,
        category=task.case.category,
        answerability=task.case.answerability,
        difficulty=task.case.difficulty,
        config_id=task.config_id,
        model=config.provider.chat_model,
        answer=response.parsed,
        retrieval_hits=hits,
        metrics=metrics,
        usage=response.usage,
        judge=judge_response.parsed if judge_response is not None else None,
        judge_model=(
            config.provider.judge_model if judge_response is not None else None
        ),
        judge_usage=judge_response.usage if judge_response is not None else None,
        http_request_count=_known_http_request_total(
            response.http_request_count,
            judge_response.http_request_count if judge_response is not None else 0,
        ),
        latency_ms=(
            response.latency_ms
            + (judge_response.latency_ms if judge_response is not None else 0)
        ),
        status="completed",
    )
    return _TaskOutput(result=result)


def _failed_case_result(
    task: _Task,
    config: ExperimentConfig,
    failure: _TaskFailure,
    *,
    cost: Decimal,
    cost_estimated: bool,
) -> CaseResult:
    cause = failure.cause
    error = (
        str(cause)
        if isinstance(cause, ProviderError)
        else f"{type(cause).__name__}: case execution failed"
    )
    return CaseResult(
        case_id=task.case.id,
        question=task.case.question,
        reference_answer=task.case.reference_answer,
        reference_evidence=task.case.reference_evidence,
        category=task.case.category,
        answerability=task.case.answerability,
        difficulty=task.case.difficulty,
        config_id=task.config_id,
        model=config.provider.chat_model,
        answer=failure.response.parsed if failure.response is not None else None,
        retrieval_hits=failure.hits,
        usage=failure.response.usage if failure.response is not None else None,
        http_request_count=_failed_http_request_count(failure),
        latency_ms=(
            failure.response.latency_ms if failure.response is not None else 0
        ),
        cost=cost,
        cost_estimated=cost_estimated,
        status="failed",
        failure_phase=failure.phase,
        error=error,
    )


def _failed_http_request_count(failure: _TaskFailure) -> int | None:
    if failure.phase == "retrieval":
        return 0
    failed_operation_count = (
        failure.cause.http_request_count
        if isinstance(failure.cause, ProviderError)
        else None
    )
    if failure.phase == "generation":
        return failed_operation_count
    generation_count = (
        failure.response.http_request_count
        if failure.response is not None
        else None
    )
    if failure.phase == "metrics":
        return generation_count
    return _known_http_request_total(generation_count, failed_operation_count)


def _known_http_request_total(*counts: int | None) -> int | None:
    if any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)


def _reconcile_failed_reservations(
    failure: _TaskFailure,
    reservations: list[Decimal],
    config: ExperimentConfig,
    ledger: BudgetLedger | None,
) -> tuple[Decimal, Decimal]:
    if ledger is None:
        return Decimal("0"), Decimal("0")
    generation = reservations[:1]
    remaining = reservations[1:]
    actual = Decimal("0")
    estimated = Decimal("0")
    if failure.phase == "retrieval":
        ledger.release_reserved(reservations)
    elif failure.phase == "generation":
        estimated = ledger.charge_reserved(generation)
        ledger.release_reserved(remaining)
    elif failure.phase == "metrics":
        if failure.response is None:
            raise ValueError("metrics failure is missing generation usage")
        actual = ledger.settle_many(
            generation,
            [(config.provider.chat_model, failure.response.usage)],
        )
        ledger.release_reserved(remaining)
    else:
        if failure.response is None:
            raise ValueError("judge failure is missing generation usage")
        actual = ledger.settle_many(
            generation,
            [(config.provider.chat_model, failure.response.usage)],
        )
        estimated = ledger.charge_reserved(remaining)
    return actual, estimated


def _summary(
    results: Sequence[CaseResult], dataset: EvaluationDataset
) -> dict[str, float]:
    completed = [result for result in results if result.status == "completed"]
    case_by_id = {case.id: case for case in dataset.cases}
    answerable = [
        result
        for result in completed
        if case_by_id[result.case_id].answerability is Answerability.ANSWERABLE
    ]
    summary: dict[str, float] = {
        "completed_cases": float(len(completed)),
        "failure_count": float(len(results) - len(completed)),
        "total_cost": float(sum((result.cost for result in results), Decimal("0"))),
        "mean_latency_ms": _mean([result.latency_ms for result in completed]),
        "p50_latency_ms": _percentile(
            [result.latency_ms for result in completed], 0.50
        ),
        "p95_latency_ms": _percentile(
            [result.latency_ms for result in completed], 0.95
        ),
    }
    for metric_name in (
        "retrieval_recall_at_k",
        "retrieval_mrr",
        "retrieval_context_hit_rate",
    ):
        summary[metric_name] = _mean(
            [result.metrics[metric_name] for result in answerable]
        )
    for metric_name in (
        "answer_exact_match",
        "answer_f1",
        "answer_semantic_similarity",
    ):
        summary[metric_name] = _mean(
            [result.metrics[metric_name] for result in answerable]
        )
    observations = [
        AbstentionObservation(
            expected_answerable=(
                case_by_id[result.case_id].answerability is Answerability.ANSWERABLE
            ),
            abstained=bool(
                result.answer
                and is_effective_abstention(
                    abstained=result.answer.abstained,
                    answer=result.answer.answer,
                )
            ),
        )
        for result in completed
    ]
    abstention = summarize_abstention(observations)
    summary.update(
        {
            "abstention_accuracy": abstention.accuracy,
            "abstention_precision": abstention.precision,
            "abstention_recall": abstention.recall,
            "abstention_f1": abstention.f1,
            "false_answer_rate": abstention.false_answer_rate,
            "over_abstention_rate": abstention.over_abstention_rate,
        }
    )
    judged = [result for result in completed if result.judge is not None]
    if judged:
        summary["judge_mean_score"] = _mean(
            [float(result.judge.score) for result in judged if result.judge is not None]
        )
        summary["judge_pass_rate"] = _mean(
            [float(result.judge.passed) for result in judged if result.judge is not None]
        )
    return summary


def planned_calls(config: ExperimentConfig, case_count: int) -> list[PlannedCall]:
    """Return every capped provider call included in one experiment plan."""

    attempts = config.provider.max_retries + 1
    calls = [
        PlannedCall(
            model=config.provider.chat_model,
            input_token_cap=(
                GENERATION_INPUT_TOKEN_CAP
                + GENERATION_OUTPUT_TOKEN_CAP
                + REPAIR_PROMPT_TOKEN_ALLOWANCE
            )
            * attempts,
            output_token_cap=GENERATION_OUTPUT_TOKEN_CAP * 2 * attempts,
            count=case_count,
            phase="generation_with_repair",
            requests_per_case=2 * attempts,
        )
    ]
    if config.provider.judge_model is not None:
        calls.append(
            PlannedCall(
                model=config.provider.judge_model,
                input_token_cap=(
                    JUDGE_INPUT_TOKEN_CAP
                    + JUDGE_OUTPUT_TOKEN_CAP
                    + REPAIR_PROMPT_TOKEN_ALLOWANCE
                )
                * attempts,
                output_token_cap=JUDGE_OUTPUT_TOKEN_CAP * 2 * attempts,
                count=case_count,
                phase="judge_with_repair",
                requests_per_case=2 * attempts,
            )
        )
    return calls


def _case_planned_calls(config: ExperimentConfig) -> list[PlannedCall]:
    return planned_calls(config, 1)


def _result_call_usages(
    result: CaseResult, config: ExperimentConfig
) -> list[tuple[str, TokenUsage]]:
    if result.usage is None:
        raise ValueError("live case result requires generation token usage")
    usages: list[tuple[str, TokenUsage]] = [
        (config.provider.chat_model, result.usage)
    ]
    if config.provider.judge_model is not None:
        if result.judge_usage is None:
            raise ValueError("live judged case result requires judge token usage")
        usages.append((config.provider.judge_model, result.judge_usage))
    return usages


def _experiment_identity(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    prompt_engine: PromptEngine,
) -> ExperimentIdentity:
    commit_sha, dirty = _git_identity()
    return ExperimentIdentity(
        name=config.name,
        mode=config.mode,
        commit_sha=commit_sha,
        dirty=dirty,
        dataset_hash=_model_hash(dataset.model_dump(mode="json")),
        prompt_hashes=prompt_engine.hashes(),
        config=config.model_dump(mode="json"),
        random_seed=config.random_seed,
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(),
    )


def _git_identity() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("pydantic", "PyYAML", "requests", "Jinja2"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _model_hash(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _config_id(retrieval: RetrievalConfig) -> str:
    return (
        f"chunk{retrieval.chunk_size}-overlap{retrieval.chunk_overlap}-"
        f"top{retrieval.top_k}-{retrieval.prompt_variant}"
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int((len(ordered) * percentile) + 0.999999) - 1)
    return ordered[index]
