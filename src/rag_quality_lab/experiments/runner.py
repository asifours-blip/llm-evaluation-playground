"""Coordinator for reproducible offline and live RAG experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from rag_quality_lab.config.loaders import load_yaml_model, validate_dataset_corpus
from rag_quality_lab.domain.models import (
    Answerability,
    CaseResult,
    Chunk,
    Document,
    EmbeddingCallRecord,
    EmbeddingPhase,
    EmbeddingResponse,
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
    estimate_tokens_upper_bound,
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
from rag_quality_lab.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    JudgeProvider,
    MeteredEmbeddingProvider,
)
from rag_quality_lab.providers.fake import FakeEmbeddingProvider, is_local_embedding_model
from rag_quality_lab.providers.openai_compatible import (
    EMBEDDING_TEXT_TOKEN_ALLOWANCE,
    GENERATED_ANSWER_EMBEDDING_BYTE_CAP,
    GENERATION_INPUT_TOKEN_CAP,
    GENERATION_OUTPUT_TOKEN_CAP,
    JUDGE_INPUT_TOKEN_CAP,
    JUDGE_OUTPUT_TOKEN_CAP,
    REPAIR_PROMPT_TOKEN_ALLOWANCE,
    ProviderError,
)
from rag_quality_lab.retrieval.index import (
    InMemoryIndex,
    chunk_document,
    embedding_cache_identity,
    load_documents,
    uncached_chunks,
)

FailurePhase = Literal["retrieval", "generation", "metrics", "judge"]
GENERATION_PHASE = "generation_with_repair"
JUDGE_PHASE = "judge_with_repair"
CASE_EMBEDDING_PHASES: tuple[EmbeddingPhase, ...] = ("embedding_query", "embedding_answer")


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
class _EmbeddingOutcome:
    """One observed embedding request batch, successful or not."""

    phase: EmbeddingPhase
    text_count: int
    input_token_upper_bound: int
    response: EmbeddingResponse | None = None
    error: Exception | None = None
    dispatched: bool = True


@dataclass(frozen=True)
class _EmbeddingCaps:
    """Per-request token upper bounds reserved for each case's live embeddings."""

    query: int
    answer: int


@dataclass(frozen=True)
class _LivePlan:
    """Structured live call plan shared by preflight and per-case reservations."""

    generation: PlannedCall
    judge: PlannedCall | None
    index_embeddings: dict[str, PlannedCall]
    query_embedding: PlannedCall | None
    answer_embedding: PlannedCall | None
    embedding_caps: _EmbeddingCaps | None

    def calls(self) -> list[PlannedCall]:
        calls = [self.generation]
        if self.judge is not None:
            calls.append(self.judge)
        calls.extend(self.index_embeddings.values())
        for call in (self.query_embedding, self.answer_embedding):
            if call is not None:
                calls.append(call)
        return calls

    def case_calls(self) -> dict[str, PlannedCall]:
        """Return the calls reserved atomically before one case is scheduled."""

        return {
            call.phase: call.model_copy(update={"count": 1})
            for call in (
                self.generation,
                self.judge,
                self.query_embedding,
                self.answer_embedding,
            )
            if call is not None
        }


@dataclass(frozen=True)
class _TaskOutput:
    result: CaseResult
    embeddings: list[_EmbeddingOutcome] = field(default_factory=list)


class _EmbeddingCallError(RuntimeError):
    """An embedding failure carrying the observed request outcome."""

    def __init__(self, outcome: _EmbeddingOutcome, cause: Exception) -> None:
        super().__init__(str(cause))
        self.outcome = outcome
        self.cause = cause


class _EmbeddingCapExceeded(ValueError):
    """Raised instead of sending embedding input larger than its reservation."""


class _TaskFailure(RuntimeError):
    """A case-local failure tagged with the pipeline phase that raised it."""

    def __init__(
        self,
        phase: FailurePhase,
        cause: Exception,
        *,
        response: ProviderResponse[StructuredAnswer] | None = None,
        hits: Sequence[RetrievalHit] = (),
        embeddings: Sequence[_EmbeddingOutcome] = (),
    ) -> None:
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause
        self.response = response
        self.hits = list(hits)
        self.embeddings = list(embeddings)


class _RecordingEmbeddingProvider:
    """Meter index-build embedding requests and refuse unplanned input."""

    def __init__(self, inner: EmbeddingProvider, *, token_cap: int) -> None:
        self.inner = inner
        self.token_cap = token_cap
        self.cache_identity = embedding_cache_identity(inner)
        self.outcome: _EmbeddingOutcome | None = None

    def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        try:
            vectors, self.outcome = _observed_embed(
                self.inner,
                texts,
                model=model,
                phase="embedding_index",
                token_cap=self.token_cap,
            )
        except _EmbeddingCallError as error:
            self.outcome = error.outcome
            raise error.cause from error
        return vectors


def run_experiment(
    config: ExperimentConfig,
    providers: ProviderBundle,
    dataset: EvaluationDataset,
) -> ExperimentRecord:
    """Run all configured case arms and persist each completed outcome.

    Live runs create the experiment record and pass the complete buffered
    preflight, including index, query, and answer embeddings, before any
    provider request is sent.
    """

    if config.provider.judge_model is not None and providers.judge is None:
        raise ValueError("judge_model requires a judge provider")
    if (
        config.mode == "live"
        and is_local_embedding_model(config.provider.embedding_model)
        and not isinstance(providers.embedding, FakeEmbeddingProvider)
    ):
        raise ValueError(
            f"embedding model {config.provider.embedding_model} is planned as a local "
            "embedding, but the embedding provider is not local"
        )
    documents = load_documents(config.knowledge_base_path)
    validate_dataset_corpus(dataset, documents)
    prompt_engine = PromptEngine()
    identity = _experiment_identity(config, dataset, prompt_engine)

    with ExperimentStore(config.database_path) as store:
        experiment_id = store.create_experiment(identity)
        status = ExperimentStatus.COMPLETED
        try:
            pricing, ledger, plan = _prepare_live_budget(
                config, dataset, documents, providers.embedding
            )
            if config.mode == "live" and ledger is None:
                summary = _summary([], dataset)
                store.finish_experiment(
                    experiment_id,
                    ExperimentStatus.BUDGET_EXCEEDED,
                    summary=summary,
                )
                return store.get_experiment(experiment_id)

            indexes = _build_indexes(
                config,
                documents,
                providers.embedding,
                store=store,
                experiment_id=experiment_id,
                ledger=ledger,
                pricing=pricing,
                plan=plan,
            )
            if indexes is None:
                budget_stopped = True
            else:
                budget_stopped = _coordinate_tasks(
                    tasks=_tasks(config, dataset, indexes, prompt_engine),
                    config=config,
                    providers=providers,
                    store=store,
                    experiment_id=experiment_id,
                    ledger=ledger,
                    pricing=pricing,
                    plan=plan,
                )
            if budget_stopped:
                status = ExperimentStatus.BUDGET_EXCEEDED
            running_record = store.get_experiment(experiment_id)
            summary = _summary(
                running_record.case_results,
                dataset,
                running_record.embedding_calls,
            )
            store.finish_experiment(experiment_id, status, summary=summary)
        except Exception:
            store.finish_experiment(experiment_id, ExperimentStatus.FAILED)
            raise
        return store.get_experiment(experiment_id)


def _prepare_live_budget(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    documents: Sequence[Document],
    embedding_provider: EmbeddingProvider,
) -> tuple[PricingConfig | None, BudgetLedger | None, _LivePlan | None]:
    if config.mode == "mock":
        return None, None, None
    if config.pricing_path is None:
        raise ValueError("live experiments require a pricing file")
    pricing = load_yaml_model(config.pricing_path, PricingConfig)
    plan = _live_plan(config, dataset, documents, embedding_provider)
    decision = preflight_budget(
        planned=plan.calls(),
        pricing=pricing,
        budget=config.budget,
    )
    if not decision.allowed:
        return pricing, None, plan
    return pricing, BudgetLedger(budget=config.budget, pricing=pricing), plan


def _build_indexes(
    config: ExperimentConfig,
    documents: Sequence[Document],
    embedding_provider: EmbeddingProvider,
    *,
    store: ExperimentStore,
    experiment_id: str,
    ledger: BudgetLedger | None,
    pricing: PricingConfig | None,
    plan: _LivePlan | None,
) -> dict[str, InMemoryIndex] | None:
    """Build every arm index; return None when the budget cannot cover it."""

    model = config.provider.embedding_model
    metered = ledger is not None and plan is not None and plan.embedding_caps is not None
    index_reservations: dict[str, Decimal] = {}
    if metered and ledger is not None and plan is not None and plan.index_embeddings:
        config_ids = list(plan.index_embeddings)
        try:
            reserved = ledger.reserve_many(
                [plan.index_embeddings[config_id] for config_id in config_ids]
            )
        except BudgetExceeded:
            return None
        index_reservations = dict(zip(config_ids, reserved, strict=True))

    indexes: dict[str, InMemoryIndex] = {}
    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        chunks = _arm_chunks(documents, retrieval)
        cache_path = _embedding_cache_path(config, config_id)
        if not metered or plan is None:
            indexes[config_id] = InMemoryIndex.from_chunks(
                chunks, embedding_provider, model=model, cache_path=cache_path
            )
            continue
        planned = plan.index_embeddings.get(config_id)
        recorder = _RecordingEmbeddingProvider(
            embedding_provider,
            token_cap=_per_request_token_bound(planned) if planned is not None else 0,
        )
        try:
            indexes[config_id] = InMemoryIndex.from_chunks(
                chunks, recorder, model=model, cache_path=cache_path
            )
        finally:
            reservation = index_reservations.pop(config_id, None)
            if recorder.outcome is not None:
                record, _ = _settle_embedding(
                    recorder.outcome,
                    reservation=reservation,
                    ledger=ledger,
                    pricing=pricing,
                    model=model,
                    config_id=config_id,
                    case_id=None,
                )
                store.record_embedding_call(experiment_id, record)
            elif reservation is not None and ledger is not None:
                ledger.release_reserved([reservation])
    return indexes


def _coordinate_tasks(
    *,
    tasks: Iterator[_Task],
    config: ExperimentConfig,
    providers: ProviderBundle,
    store: ExperimentStore,
    experiment_id: str,
    ledger: BudgetLedger | None,
    pricing: PricingConfig | None,
    plan: _LivePlan | None,
) -> bool:
    pending: dict[Future[_TaskOutput], tuple[_Task, dict[str, Decimal]]] = {}
    no_more_tasks = False
    budget_stopped = False
    caps = plan.embedding_caps if plan is not None else None
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
                reservations: dict[str, Decimal] = {}
                if ledger is not None and plan is not None:
                    case_calls = plan.case_calls()
                    try:
                        reserved = ledger.reserve_many(list(case_calls.values()))
                    except BudgetExceeded:
                        budget_stopped = True
                        break
                    reservations = dict(zip(case_calls, reserved, strict=True))
                future = executor.submit(_evaluate_task, task, config, providers, caps)
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
                    embeddings = output.embeddings
                except _TaskFailure as failure:
                    if _record_case_embeddings(
                        failure.embeddings,
                        reservations,
                        task=task,
                        config=config,
                        store=store,
                        experiment_id=experiment_id,
                        ledger=ledger,
                        pricing=pricing,
                    ):
                        budget_stopped = True
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
                if _record_case_embeddings(
                    embeddings,
                    reservations,
                    task=task,
                    config=config,
                    store=store,
                    experiment_id=experiment_id,
                    ledger=ledger,
                    pricing=pricing,
                ):
                    budget_stopped = True
                if ledger is not None and pricing is not None:
                    call_usages = _result_call_usages(result, config)
                    try:
                        actual_cost = ledger.settle_many(
                            _chat_reservations(reservations), call_usages
                        )
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


def _record_case_embeddings(
    outcomes: Sequence[_EmbeddingOutcome],
    reservations: dict[str, Decimal],
    *,
    task: _Task,
    config: ExperimentConfig,
    store: ExperimentStore,
    experiment_id: str,
    ledger: BudgetLedger | None,
    pricing: PricingConfig | None,
) -> bool:
    """Settle and persist planned per-case embeddings; release unreached ones."""

    budget_stopped = False
    by_phase = {outcome.phase: outcome for outcome in outcomes}
    for phase in CASE_EMBEDDING_PHASES:
        if phase not in reservations:
            continue
        reservation = reservations[phase]
        outcome = by_phase.get(phase)
        if outcome is None:
            if ledger is not None:
                ledger.release_reserved([reservation])
            continue
        record, exceeded = _settle_embedding(
            outcome,
            reservation=reservation,
            ledger=ledger,
            pricing=pricing,
            model=config.provider.embedding_model,
            config_id=task.config_id,
            case_id=task.case.id,
        )
        store.record_embedding_call(experiment_id, record)
        budget_stopped = budget_stopped or exceeded
    return budget_stopped


def _settle_embedding(
    outcome: _EmbeddingOutcome,
    *,
    reservation: Decimal | None,
    ledger: BudgetLedger | None,
    pricing: PricingConfig | None,
    model: str,
    config_id: str,
    case_id: str | None,
) -> tuple[EmbeddingCallRecord, bool]:
    """Settle one embedding outcome and return its record plus a budget-stop flag."""

    usage = outcome.response.usage if outcome.response is not None else None
    cost = Decimal("0")
    estimated = False
    exceeded = False
    if ledger is not None and reservation is not None:
        if not outcome.dispatched:
            ledger.release_reserved([reservation])
        elif usage is not None:
            try:
                cost = ledger.settle_many([reservation], [(model, usage)])
            except BudgetExceeded:
                if pricing is None:
                    raise
                cost = calculate_actual_cost(usage, pricing.models[model])
                exceeded = True
        else:
            cost = ledger.charge_reserved([reservation])
            estimated = True
    return (
        EmbeddingCallRecord(
            phase=outcome.phase,
            config_id=config_id,
            case_id=case_id,
            model=model,
            text_count=outcome.text_count,
            input_token_upper_bound=outcome.input_token_upper_bound,
            usage=usage,
            http_request_count=_embedding_http_request_count(outcome),
            cost=cost,
            cost_estimated=estimated,
            status="completed" if outcome.error is None else "failed",
            error=_embedding_error(outcome.error),
        ),
        exceeded,
    )


def _embedding_http_request_count(outcome: _EmbeddingOutcome) -> int | None:
    if not outcome.dispatched:
        return 0
    if outcome.response is not None:
        return outcome.response.http_request_count
    if isinstance(outcome.error, ProviderError):
        return outcome.error.http_request_count
    return None


def _embedding_error(error: Exception | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, ProviderError | _EmbeddingCapExceeded):
        return str(error)
    return f"{type(error).__name__}: embedding request failed"


def _observed_embed(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    model: str | None,
    phase: EmbeddingPhase,
    token_cap: int | None,
) -> tuple[list[list[float]], _EmbeddingOutcome]:
    """Embed texts within an optional reserved token cap and observe the request."""

    bound = _embedding_token_bound(texts)
    if token_cap is not None and bound > token_cap:
        cap_error = _EmbeddingCapExceeded(
            f"{phase} input upper bound {bound} exceeds the reserved cap {token_cap}"
        )
        raise _EmbeddingCallError(
            _EmbeddingOutcome(
                phase=phase,
                text_count=len(texts),
                input_token_upper_bound=bound,
                error=cap_error,
                dispatched=False,
            ),
            cap_error,
        )
    try:
        if isinstance(provider, MeteredEmbeddingProvider):
            response = provider.embed_with_metadata(texts, model=model)
        else:
            response = EmbeddingResponse(
                vectors=provider.embed(texts, model=model), model=model
            )
    except Exception as error:
        not_dispatched = (
            isinstance(error, ProviderError) and error.http_request_count == 0
        )
        raise _EmbeddingCallError(
            _EmbeddingOutcome(
                phase=phase,
                text_count=len(texts),
                input_token_upper_bound=bound,
                error=error,
                dispatched=not not_dispatched,
            ),
            error,
        ) from error
    outcome = _EmbeddingOutcome(
        phase=phase,
        text_count=len(texts),
        input_token_upper_bound=bound,
        response=response,
    )
    if len(response.vectors) != len(texts):
        count_error = ValueError("embedding provider returned an unexpected vector count")
        raise _EmbeddingCallError(
            _EmbeddingOutcome(
                phase=phase,
                text_count=len(texts),
                input_token_upper_bound=bound,
                response=response,
                error=count_error,
            ),
            count_error,
        )
    return response.vectors, outcome


def _tasks(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    indexes: dict[str, InMemoryIndex],
    prompt_engine: PromptEngine,
) -> Iterator[_Task]:
    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        for case in dataset.cases:
            yield _Task(
                case=case,
                retrieval=retrieval,
                config_id=config_id,
                index=indexes[config_id],
                instructions=prompt_engine.instructions(retrieval.prompt_variant),
            )


def _arm_chunks(
    documents: Sequence[Document], retrieval: RetrievalConfig
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(
            document.id,
            document.text,
            chunk_size=retrieval.chunk_size,
            chunk_overlap=retrieval.chunk_overlap,
        )
    ]


def _embedding_cache_path(config: ExperimentConfig, config_id: str) -> Path:
    return config.artifact_dir / f"{config_id}.embeddings.json"


def _evaluate_task(
    task: _Task,
    config: ExperimentConfig,
    providers: ProviderBundle,
    caps: _EmbeddingCaps | None = None,
) -> _TaskOutput:
    embeddings: list[_EmbeddingOutcome] = []
    try:
        query_vectors, query_outcome = _observed_embed(
            providers.embedding,
            [task.case.question],
            model=config.provider.embedding_model,
            phase="embedding_query",
            token_cap=caps.query if caps is not None else None,
        )
        embeddings.append(query_outcome)
        hits = task.index.rank(query_vectors[0], top_k=task.retrieval.top_k)
    except _EmbeddingCallError as error:
        embeddings.append(error.outcome)
        raise _TaskFailure("retrieval", error.cause, embeddings=embeddings) from error
    except Exception as error:
        raise _TaskFailure("retrieval", error, embeddings=embeddings) from error
    contexts = [f"[{hit.chunk.id}]\n{hit.chunk.text}" for hit in hits]
    try:
        response = providers.chat.answer(
            task.case.question,
            contexts,
            model=config.provider.chat_model,
            instructions=task.instructions,
        )
    except Exception as error:
        raise _TaskFailure(
            "generation", error, hits=hits, embeddings=embeddings
        ) from error
    try:
        answer_vectors, answer_outcome = _observed_embed(
            providers.embedding,
            [response.parsed.answer, task.case.reference_answer],
            model=config.provider.embedding_model,
            phase="embedding_answer",
            token_cap=caps.answer if caps is not None else None,
        )
        embeddings.append(answer_outcome)
    except _EmbeddingCallError as error:
        embeddings.append(error.outcome)
        raise _TaskFailure(
            "metrics", error.cause, response=response, hits=hits, embeddings=embeddings
        ) from error
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
            raise _TaskFailure(
                "judge", error, response=response, hits=hits, embeddings=embeddings
            ) from error
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
    return _TaskOutput(result=result, embeddings=embeddings)


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
    reservations: dict[str, Decimal],
    config: ExperimentConfig,
    ledger: BudgetLedger | None,
) -> tuple[Decimal, Decimal]:
    """Reconcile generation and judge reservations for one failed case."""

    if ledger is None:
        return Decimal("0"), Decimal("0")
    generation = _chat_reservations(reservations)[:1]
    remaining = _chat_reservations(reservations)[1:]
    actual = Decimal("0")
    estimated = Decimal("0")
    if failure.phase == "retrieval":
        ledger.release_reserved(generation + remaining)
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


def _chat_reservations(reservations: dict[str, Decimal]) -> list[Decimal]:
    """Return generation then judge reservations, matching settlement order."""

    return [
        reservations[phase]
        for phase in (GENERATION_PHASE, JUDGE_PHASE)
        if phase in reservations
    ]


def _summary(
    results: Sequence[CaseResult],
    dataset: EvaluationDataset,
    embedding_calls: Sequence[EmbeddingCallRecord] = (),
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
    if embedding_calls:
        case_cost = sum((result.cost for result in results), Decimal("0"))
        embedding_cost = sum((call.cost for call in embedding_calls), Decimal("0"))
        summary["embedding_cost"] = float(embedding_cost)
        summary["total_cost"] = float(case_cost + embedding_cost)
        request_counts = [call.http_request_count for call in embedding_calls]
        if all(count is not None for count in request_counts):
            summary["embedding_http_request_count"] = float(
                sum(count for count in request_counts if count is not None)
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


def planned_calls(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    documents: Sequence[Document],
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[PlannedCall]:
    """Return every capped provider call included in one experiment plan.

    Remote embedding calls are planned per phase: one index batch per arm,
    one query batch per case, and one answer batch per case. Without an
    ``embedding_provider`` every chunk is counted; with one, only chunks
    whose cache entry matches provider, model, text, and chunking are deducted.
    The local ``fake-hash`` embedding sends no requests and is not planned.
    """

    return _live_plan(config, dataset, documents, embedding_provider).calls()


def _live_plan(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    documents: Sequence[Document],
    embedding_provider: EmbeddingProvider | None,
) -> _LivePlan:
    case_count = len(dataset.cases) * len(config.retrieval)
    attempts = config.provider.max_retries + 1
    generation = PlannedCall(
        model=config.provider.chat_model,
        input_token_cap=(
            GENERATION_INPUT_TOKEN_CAP
            + GENERATION_OUTPUT_TOKEN_CAP
            + REPAIR_PROMPT_TOKEN_ALLOWANCE
        )
        * attempts,
        output_token_cap=GENERATION_OUTPUT_TOKEN_CAP * 2 * attempts,
        count=case_count,
        phase=GENERATION_PHASE,
        requests_per_case=2 * attempts,
    )
    judge = None
    if config.provider.judge_model is not None:
        judge = PlannedCall(
            model=config.provider.judge_model,
            input_token_cap=(
                JUDGE_INPUT_TOKEN_CAP
                + JUDGE_OUTPUT_TOKEN_CAP
                + REPAIR_PROMPT_TOKEN_ALLOWANCE
            )
            * attempts,
            output_token_cap=JUDGE_OUTPUT_TOKEN_CAP * 2 * attempts,
            count=case_count,
            phase=JUDGE_PHASE,
            requests_per_case=2 * attempts,
        )
    model = config.provider.embedding_model
    if is_local_embedding_model(model):
        return _LivePlan(
            generation=generation,
            judge=judge,
            index_embeddings={},
            query_embedding=None,
            answer_embedding=None,
            embedding_caps=None,
        )

    index_embeddings: dict[str, PlannedCall] = {}
    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        chunks = _arm_chunks(documents, retrieval)
        if embedding_provider is not None:
            chunks = uncached_chunks(
                chunks,
                embedding_provider,
                model=model,
                cache_path=_embedding_cache_path(config, config_id),
            )
        if chunks:
            index_embeddings[config_id] = PlannedCall(
                model=model,
                input_token_cap=_embedding_token_bound([chunk.text for chunk in chunks])
                * attempts,
                output_token_cap=0,
                count=1,
                phase="embedding_index",
                requests_per_case=attempts,
            )
    caps = _EmbeddingCaps(
        query=max(_embedding_token_bound([case.question]) for case in dataset.cases),
        answer=max(
            GENERATED_ANSWER_EMBEDDING_BYTE_CAP
            + EMBEDDING_TEXT_TOKEN_ALLOWANCE
            + _embedding_token_bound([case.reference_answer])
            for case in dataset.cases
        ),
    )
    return _LivePlan(
        generation=generation,
        judge=judge,
        index_embeddings=index_embeddings,
        query_embedding=PlannedCall(
            model=model,
            input_token_cap=caps.query * attempts,
            output_token_cap=0,
            count=case_count,
            phase="embedding_query",
            requests_per_case=attempts,
        ),
        answer_embedding=PlannedCall(
            model=model,
            input_token_cap=caps.answer * attempts,
            output_token_cap=0,
            count=case_count,
            phase="embedding_answer",
            requests_per_case=attempts,
        ),
        embedding_caps=caps,
    )


def _embedding_token_bound(texts: Sequence[str]) -> int:
    """Conservative token upper bound for one embedding request."""

    return sum(
        estimate_tokens_upper_bound(text) + EMBEDDING_TEXT_TOKEN_ALLOWANCE
        for text in texts
    )


def _per_request_token_bound(call: PlannedCall) -> int:
    """Undo the retry multiplier applied to a planned embedding input cap."""

    return call.input_token_cap // call.requests_per_case


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
