"""Coordinator for reproducible offline and live RAG experiments."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

from rag_quality_lab.config.holdout import (
    HoldoutStatus,
    holdout_lock_path,
    load_holdout_lock,
    verify_holdout,
)
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
    LedgerEntry,
    PricingConfig,
    ProviderResponse,
    RetrievalConfig,
    RetrievalHit,
    StructuredAnswer,
    TokenUsage,
    canonical_hash,
)
from rag_quality_lab.experiments.budget import (
    MAX_PRICING_AGE_DAYS,
    BudgetExceeded,
    BudgetLedger,
    PlannedCall,
    calculate_actual_cost,
    estimate_tokens_upper_bound,
    preflight_budget,
)
from rag_quality_lab.experiments.compare import canonical_config, diff_paths
from rag_quality_lab.experiments.liveness import LeaseHeartbeat
from rag_quality_lab.experiments.store import (
    DispatchJournal,
    ExperimentStore,
    SettledState,
    Settlement,
    resume_refusal,
)
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
from rag_quality_lab.prompts.engine import PROMPT_VERSION, PromptEngine
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
    embedding_text_token_bound,
    split_embedding_batches,
)
from rag_quality_lab.retrieval.bm25 import BM25Index
from rag_quality_lab.retrieval.index import (
    InMemoryIndex,
    chunk_document,
    embedding_cache_identity,
    load_documents,
    uncached_chunks,
)

FailurePhase = Literal["retrieval", "generation", "metrics", "judge"]
Retriever = InMemoryIndex | BM25Index
StopReason = Literal["budget", "cancelled"] | None
DispatchMarker = Callable[[str], None]
GENERATION_PHASE = "generation_with_repair"
JUDGE_PHASE = "judge_with_repair"
CASE_EMBEDDING_PHASES: tuple[EmbeddingPhase, ...] = ("embedding_query", "embedding_answer")
FINAL_STATUS: dict[StopReason, ExperimentStatus] = {
    None: ExperimentStatus.COMPLETED,
    "budget": ExperimentStatus.BUDGET_EXCEEDED,
    "cancelled": ExperimentStatus.CANCELLED,
}


class ResumeRefused(ValueError):
    """An experiment cannot be resumed; no request was sent and no state changed."""


class _TaskCancelled(RuntimeError):
    """Raised instead of dispatching a phase once cancellation is observed."""

    def __init__(self, phase: str) -> None:
        super().__init__(f"cancelled before {phase} was sent")
        self.phase = phase


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
    index: Retriever
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
    batch_index: int = 0
    batch_count: int = 1

    @property
    def key(self) -> str:
        return _batch_key(self.phase, self.batch_index)


@dataclass(frozen=True)
class _PhaseBatches:
    """Planned request batches of one per-case embedding phase.

    ``ranges`` split the phase's texts into requests and ``caps`` are the
    reserved token upper bounds of each request. They are derived from the
    planned per-text bounds, not the actual texts, so the number of requests a
    case sends always equals the number planned and reserved.
    """

    phase: EmbeddingPhase
    ranges: tuple[range, ...]
    caps: tuple[int, ...]

    @classmethod
    def plan(
        cls, phase: EmbeddingPhase, bounds: Sequence[int], config: ExperimentConfig
    ) -> _PhaseBatches:
        ranges = split_embedding_batches(
            bounds,
            max_inputs=config.provider.embedding_batch_size,
            max_tokens=config.provider.embedding_batch_token_limit,
        )
        return cls(
            phase=phase,
            ranges=tuple(ranges),
            caps=tuple(sum(bounds[index] for index in batch) for batch in ranges),
        )


@dataclass(frozen=True)
class _CaseBatches:
    """Planned embedding batches of every live case arm."""

    query: _PhaseBatches
    answer: _PhaseBatches


@dataclass(frozen=True)
class _LivePlan:
    """Structured live call plan shared by preflight and per-case reservations.

    Remote embedding phases hold one planned call per request batch, keyed
    ``phase`` for the first batch and ``phase#N`` for later ones.
    """

    generation: PlannedCall
    judge: PlannedCall | None
    index_embeddings: dict[str, list[PlannedCall]]
    query_embedding: list[PlannedCall]
    answer_embedding: list[PlannedCall]
    case_batches: _CaseBatches | None

    def calls(self) -> list[PlannedCall]:
        calls = [self.generation]
        if self.judge is not None:
            calls.append(self.judge)
        for batches in self.index_embeddings.values():
            calls.extend(batches)
        calls.extend(self.query_embedding)
        calls.extend(self.answer_embedding)
        return calls

    def case_calls(self, retriever: str) -> dict[str, PlannedCall]:
        """Return the calls reserved atomically before one case is scheduled.

        A BM25 arm embeds no query, so it reserves no query batches.
        """

        calls = [self.generation]
        if self.judge is not None:
            calls.append(self.judge)
        if retriever == "embedding":
            calls.extend(self.query_embedding)
        calls.extend(self.answer_embedding)
        return {call.phase: call.model_copy(update={"count": 1}) for call in calls}


@dataclass(frozen=True)
class _TaskOutput:
    result: CaseResult
    embeddings: list[_EmbeddingOutcome] = field(default_factory=list)


@dataclass(frozen=True)
class _Claim:
    """One scheduled task with its in-memory and journaled reservations."""

    task: _Task
    reservations: dict[str, Decimal]
    entry_ids: dict[str, int]


@dataclass(frozen=True)
class _PhaseCharge:
    """Final ledger outcome of one reserved provider phase."""

    state: SettledState
    charged: Decimal
    estimated: bool = False


@dataclass(frozen=True)
class _Session:
    """Everything one run or resume attempt needs to execute and settle tasks."""

    config: ExperimentConfig
    providers: ProviderBundle
    dataset: EvaluationDataset
    documents: Sequence[Document]
    prompt_engine: PromptEngine
    store: ExperimentStore
    experiment_id: str
    ledger: BudgetLedger | None
    pricing: PricingConfig | None
    plan: _LivePlan | None
    journal: DispatchJournal | None


class _EmbeddingCallError(RuntimeError):
    """An embedding failure carrying every observed request batch outcome."""

    def __init__(self, outcomes: Sequence[_EmbeddingOutcome], cause: Exception) -> None:
        super().__init__(str(cause))
        self.outcomes = list(outcomes)
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
    """Meter index-build embedding batches and refuse unplanned input."""

    def __init__(
        self,
        inner: EmbeddingProvider,
        *,
        planned: Sequence[PlannedCall],
        config: ExperimentConfig,
        mark: DispatchMarker | None = None,
    ) -> None:
        self.inner = inner
        self.planned = list(planned)
        self.config = config
        self.mark = mark
        self.cache_identity = embedding_cache_identity(inner)
        self.outcomes: list[_EmbeddingOutcome] = []

    def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        bounds = [embedding_text_token_bound(text) for text in texts]
        planned = _PhaseBatches.plan("embedding_index", bounds, self.config)
        if len(planned.ranges) != len(self.planned):
            raise ValueError(
                f"index embedding needs {len(planned.ranges)} batches but "
                f"{len(self.planned)} were planned"
            )
        batches = _PhaseBatches(
            phase="embedding_index",
            ranges=planned.ranges,
            caps=tuple(_per_request_token_bound(call) for call in self.planned),
        )
        try:
            vectors, self.outcomes = _observed_embed_batches(
                self.inner, texts, model=model, batches=batches, mark=self.mark
            )
        except _EmbeddingCallError as error:
            self.outcomes = error.outcomes
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
    provider request is sent. The record freezes the configuration, dataset,
    corpus, prompt, and pricing identity so an interrupted run can be resumed
    with :func:`resume_experiment`.
    """

    _validate_providers(config, providers)
    documents = load_documents(config.knowledge_base_path)
    validate_dataset_corpus(dataset, documents)
    holdout = _verified_holdout(config, dataset)
    prompt_engine = PromptEngine()
    pricing = _load_pricing(config)
    identity = _experiment_identity(
        config, dataset, prompt_engine, documents, pricing, holdout
    )
    dataset = dataset.select_splits(config.splits)

    with ExperimentStore(config.database_path) as store:
        store.reap_orphans()
        experiment_id = store.create_experiment(
            identity,
            planned_task_count=len(dataset.cases) * len(config.retrieval),
            metadata=_attempt_metadata(identity),
        )
        with ExitStack() as resources:
            _hold_lease(resources, store, experiment_id)
            try:
                ledger, plan = _prepare_live_budget(
                    config, dataset, documents, providers.embedding, pricing
                )
                if config.mode == "live" and ledger is None:
                    store.finish_experiment(
                        experiment_id,
                        ExperimentStatus.BUDGET_EXCEEDED,
                        summary=_summary([], dataset),
                    )
                    return store.get_experiment(experiment_id)
                session = _Session(
                    config=config,
                    providers=providers,
                    dataset=dataset,
                    documents=documents,
                    prompt_engine=prompt_engine,
                    store=store,
                    experiment_id=experiment_id,
                    ledger=ledger,
                    pricing=pricing,
                    plan=plan,
                    journal=_open_journal(resources, store),
                )
                stop = _execute(session, skip=set())
                _finish(store, experiment_id, dataset, stop)
            except Exception:
                store.finish_experiment(experiment_id, ExperimentStatus.FAILED)
                raise
            except BaseException:
                store.mark_interrupted(experiment_id)
                raise
        return store.get_experiment(experiment_id)


def resume_experiment(
    experiment_id: str,
    config: ExperimentConfig,
    providers: ProviderBundle,
    dataset: EvaluationDataset,
    *,
    retry_unknown: bool = False,
    on_date: date | None = None,
) -> ExperimentRecord:
    """Continue an INTERRUPTED experiment from its persisted checkpoint.

    The current configuration, dataset, corpus, prompts, and pricing must
    match the identity frozen when the experiment started. Case arms with a
    persisted outcome are skipped. Case arms whose calls were sent but never
    settled are charged at their reservation and skipped unless
    ``retry_unknown`` is set, in which case they run again and are charged
    again. The budget ledger continues from the persisted spend, and the
    remaining plan must pass preflight against the unspent budget. Because
    resuming spends money again, the frozen pricing must still be fresh on
    ``on_date`` (default: today).
    """

    _validate_providers(config, providers)
    documents = load_documents(config.knowledge_base_path)
    validate_dataset_corpus(dataset, documents)
    holdout = _verified_holdout(config, dataset)
    prompt_engine = PromptEngine()
    pricing = _load_pricing(config)
    current = _experiment_identity(
        config, dataset, prompt_engine, documents, pricing, holdout
    )
    dataset = dataset.select_splits(config.splits)

    with ExperimentStore(config.database_path) as store:
        store.reap_orphans()
        record = store.get_experiment(experiment_id)
        _require_resumable(experiment_id, record, retry_unknown=retry_unknown)
        changes = identity_changes(record.identity, current)
        if changes:
            raise ResumeRefused(
                f"cannot resume experiment {experiment_id}: inputs differ from its "
                f"frozen identity: {', '.join(changes)}"
            )
        done = {
            _case_task_key(config_id, case_id)
            for case_id, config_id in store.recorded_case_keys(experiment_id)
        }
        unknown = store.unknown_task_keys(experiment_id)
        skip = done if retry_unknown else done | unknown
        remaining = [
            (config_id, case.id)
            for config_id in (_config_id(retrieval) for retrieval in config.retrieval)
            for case in dataset.cases
            if _case_task_key(config_id, case.id) not in skip
        ]
        arms = {config_id for config_id, _ in remaining}
        _refuse_unknown_index_rebuild(
            experiment_id, config, documents, providers.embedding, arms, unknown, retry_unknown
        )
        ledger: BudgetLedger | None = None
        plan: _LivePlan | None = None
        if config.mode == "live" and remaining:
            if pricing is None:
                raise ValueError("live experiments require a pricing file")
            evaluation_date = on_date or date.today()
            if pricing.is_stale(evaluation_date, max_age_days=MAX_PRICING_AGE_DAYS):
                raise ResumeRefused(
                    f"cannot resume experiment {experiment_id}: its pricing snapshot "
                    f"was verified on {pricing.verified_at.isoformat()} and is older "
                    f"than the {MAX_PRICING_AGE_DAYS} days pricing stays valid on "
                    f"{evaluation_date.isoformat()}; pricing is frozen in the experiment "
                    "identity, so start a new experiment to use new pricing"
                )
            spent = store.ledger_spent(experiment_id)
            plan = _live_plan(
                config,
                dataset,
                documents,
                providers.embedding,
                remaining=remaining,
            )
            decision = preflight_budget(
                planned=plan.calls(),
                pricing=pricing,
                budget=config.budget,
                on_date=evaluation_date,
                spent=spent,
            )
            if not decision.allowed:
                raise ResumeRefused(
                    f"cannot resume experiment {experiment_id}: {decision.reason}; "
                    f"spent {spent} of the {config.budget.hard_limit} "
                    f"{config.budget.currency} hard budget, remaining plan needs "
                    f"{decision.buffered_cost} against a threshold of {decision.threshold}"
                )
            ledger = BudgetLedger(budget=config.budget, pricing=pricing, spent=spent)
        try:
            store.claim_for_resume(
                experiment_id,
                retry_unknown=retry_unknown,
                metadata={
                    **_attempt_metadata(current),
                    "retry_unknown": retry_unknown,
                },
            )
        except ValueError as error:
            raise ResumeRefused(str(error)) from error

        with ExitStack() as resources:
            _hold_lease(resources, store, experiment_id)
            try:
                stop: StopReason = None
                if remaining:
                    session = _Session(
                        config=config,
                        providers=providers,
                        dataset=dataset,
                        documents=documents,
                        prompt_engine=prompt_engine,
                        store=store,
                        experiment_id=experiment_id,
                        ledger=ledger,
                        pricing=pricing,
                        plan=plan,
                        journal=_open_journal(resources, store),
                    )
                    stop = _execute(session, skip=skip)
                _finish(store, experiment_id, dataset, stop)
            except Exception:
                store.finish_experiment(experiment_id, ExperimentStatus.FAILED)
                raise
            except BaseException:
                store.mark_interrupted(experiment_id)
                raise
        return store.get_experiment(experiment_id)


def identity_changes(
    frozen: ExperimentIdentity, current: ExperimentIdentity
) -> list[str]:
    """Name every resume-relevant input that differs from the frozen identity."""

    changes: list[str] = []
    if frozen.mode != current.mode:
        changes.append("mode")
    if frozen.dataset_hash != current.dataset_hash:
        changes.append("dataset content (dataset_hash)")
    if frozen.corpus_hash != current.corpus_hash:
        changes.append("knowledge-base corpus content (corpus_hash)")
    if (
        frozen.prompt_version != current.prompt_version
        or frozen.prompt_hashes != current.prompt_hashes
    ):
        changes.append("prompt version or templates (prompt_hashes)")
    # Stored configs are re-validated so fields added later take their
    # defaults instead of reading as changes.
    changes.extend(
        f"config.{path}"
        for path in diff_paths(
            canonical_config(frozen.config), canonical_config(current.config)
        )
    )
    changes.extend(
        f"pricing.{path}" if path else "pricing"
        for path in diff_paths(frozen.pricing_snapshot, current.pricing_snapshot)
    )
    return changes


def _require_resumable(
    experiment_id: str, record: ExperimentRecord, *, retry_unknown: bool
) -> None:
    resumable = {ExperimentStatus.INTERRUPTED}
    if retry_unknown:
        resumable.add(ExperimentStatus.INCOMPLETE)
    if record.status not in resumable:
        raise ResumeRefused(resume_refusal(experiment_id, record.status))
    if record.identity.corpus_hash is None or record.identity.prompt_version is None:
        raise ResumeRefused(
            f"cannot resume experiment {experiment_id}: it was recorded before "
            "resumable identities existed and has no frozen corpus or pricing snapshot"
        )


def _refuse_unknown_index_rebuild(
    experiment_id: str,
    config: ExperimentConfig,
    documents: Sequence[Document],
    embedding_provider: EmbeddingProvider,
    arms: set[str],
    unknown: set[str],
    retry_unknown: bool,
) -> None:
    """Refuse to silently re-send an index embedding whose outcome is unknown."""

    if retry_unknown:
        return
    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        if retrieval.retriever != "embedding":
            continue
        if config_id not in arms or _index_task_key(config_id) not in unknown:
            continue
        if uncached_chunks(
            _arm_chunks(documents, retrieval),
            embedding_provider,
            model=config.provider.embedding_model,
            cache_path=_embedding_cache_path(config, config_id),
        ):
            raise ResumeRefused(
                f"cannot resume experiment {experiment_id}: the index embedding for "
                f"{config_id} has an unknown outcome and its cache is incomplete; "
                "pass --retry-unknown to send it again at additional cost"
            )


def _verified_holdout(config: ExperimentConfig, dataset: EvaluationDataset) -> HoldoutStatus:
    """Refuse a dataset whose frozen holdout changed without a new version."""

    return verify_holdout(dataset, load_holdout_lock(holdout_lock_path(config.dataset_path)))


def _validate_providers(config: ExperimentConfig, providers: ProviderBundle) -> None:
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
    for setting in ("embedding_batch_size", "embedding_batch_token_limit"):
        provider_value = getattr(providers.embedding, setting, None)
        if provider_value is not None and provider_value != getattr(config.provider, setting):
            # The runner plans and reserves batches from the configuration, so
            # a provider that splits differently would send unplanned requests.
            raise ValueError(
                f"embedding provider {setting}={provider_value} differs from "
                f"provider.{setting}={getattr(config.provider, setting)}"
            )


def _load_pricing(config: ExperimentConfig) -> PricingConfig | None:
    if config.pricing_path is None:
        if config.mode == "live":
            raise ValueError("live experiments require a pricing file")
        return None
    return load_yaml_model(config.pricing_path, PricingConfig)


def _hold_lease(resources: ExitStack, store: ExperimentStore, experiment_id: str) -> None:
    resources.enter_context(
        LeaseHeartbeat(store.path, experiment_id, store.lease_token(experiment_id))
    )


def _open_journal(resources: ExitStack, store: ExperimentStore) -> DispatchJournal:
    return resources.enter_context(store.dispatch_journal())


def _finish(
    store: ExperimentStore,
    experiment_id: str,
    dataset: EvaluationDataset,
    stop: StopReason,
) -> None:
    record = store.get_experiment(experiment_id)
    status = FINAL_STATUS[stop]
    unresolved = {
        call.task_key for call in record.unknown_calls if call.case_id is not None
    } - {
        _case_task_key(result.config_id, result.case_id) for result in record.case_results
    }
    if status is ExperimentStatus.COMPLETED and unresolved:
        # Every other case arm finished; unknown outcomes still need a retry.
        status = ExperimentStatus.INCOMPLETE
    store.finish_experiment(
        experiment_id,
        status,
        summary=_summary(
            record.case_results,
            dataset,
            record.embedding_calls,
            record.unknown_calls,
        ),
    )


def _attempt_metadata(identity: ExperimentIdentity) -> dict[str, object]:
    return {
        "commit_sha": identity.commit_sha,
        "dirty": identity.dirty,
        "python_version": identity.python_version,
    }


def _execute(session: _Session, *, skip: set[str]) -> StopReason:
    """Build the indexes still needed and run every task not in ``skip``."""

    if session.store.cancel_requested(session.experiment_id):
        return "cancelled"
    arms = {
        _config_id(retrieval)
        for retrieval in session.config.retrieval
        for case in session.dataset.cases
        if _case_task_key(_config_id(retrieval), case.id) not in skip
    }
    indexes = _build_indexes(session, arms)
    if indexes is None:
        return "budget"
    return _coordinate_tasks(
        _tasks(session.config, session.dataset, indexes, session.prompt_engine, skip),
        session,
    )


def _prepare_live_budget(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    documents: Sequence[Document],
    embedding_provider: EmbeddingProvider,
    pricing: PricingConfig | None,
) -> tuple[BudgetLedger | None, _LivePlan | None]:
    if config.mode == "mock":
        return None, None
    if pricing is None:
        raise ValueError("live experiments require a pricing file")
    plan = _live_plan(config, dataset, documents, embedding_provider)
    decision = preflight_budget(
        planned=plan.calls(),
        pricing=pricing,
        budget=config.budget,
    )
    if not decision.allowed:
        return None, plan
    return BudgetLedger(budget=config.budget, pricing=pricing), plan


def _build_indexes(session: _Session, arms: set[str]) -> dict[str, Retriever] | None:
    """Build the indexes of ``arms``; return None when the budget cannot cover it.

    BM25 arms build a local lexical index and never call the embedding
    provider. Dense chunks whose cache entry matches provider, model, text,
    and chunking are never embedded again, so a resumed run reuses a verified
    cached index. Each planned index batch is reserved, journaled as sent, and
    settled on its own: sent batches are charged even when a later one fails,
    and batches never sent are released.
    """

    config = session.config
    ledger = session.ledger
    plan = session.plan
    model = config.provider.embedding_model
    metered = ledger is not None and plan is not None and plan.case_batches is not None
    index_reservations: dict[str, dict[str, Decimal]] = {}
    entry_ids: dict[str, dict[str, int]] = {}
    if metered and ledger is not None and plan is not None and plan.index_embeddings:
        config_ids = [config_id for config_id in plan.index_embeddings if config_id in arms]
        planned_batches = [
            call for config_id in config_ids for call in plan.index_embeddings[config_id]
        ]
        try:
            reserved = iter(ledger.reserve_many(planned_batches))
        except BudgetExceeded:
            return None
        for config_id in config_ids:
            index_reservations[config_id] = {
                call.phase: next(reserved) for call in plan.index_embeddings[config_id]
            }
            entry_ids[config_id] = session.store.reserve_entries(
                session.experiment_id,
                task_key=_index_task_key(config_id),
                config_id=config_id,
                case_id=None,
                reservations=index_reservations[config_id],
            )

    indexes: dict[str, Retriever] = {}
    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        if config_id not in arms:
            continue
        chunks = _arm_chunks(session.documents, retrieval)
        if retrieval.retriever == "bm25":
            indexes[config_id] = BM25Index.from_chunks(chunks)
            continue
        cache_path = _embedding_cache_path(config, config_id)
        if not metered or plan is None:
            indexes[config_id] = InMemoryIndex.from_chunks(
                chunks, session.providers.embedding, model=model, cache_path=cache_path
            )
            continue
        arm_entries = entry_ids.get(config_id, {})
        recorder = _RecordingEmbeddingProvider(
            session.providers.embedding,
            planned=plan.index_embeddings.get(config_id, []),
            config=config,
            mark=_dispatch_gate(
                session.journal, session.experiment_id, arm_entries, check_cancel=False
            ),
        )
        try:
            indexes[config_id] = InMemoryIndex.from_chunks(
                chunks, recorder, model=model, cache_path=cache_path
            )
        finally:
            reservations = index_reservations.pop(config_id, {})
            for outcome in recorder.outcomes:
                record, _ = _settle_embedding(
                    outcome,
                    reservation=reservations.pop(outcome.key, None),
                    ledger=ledger,
                    pricing=session.pricing,
                    model=model,
                    config_id=config_id,
                    case_id=None,
                )
                entry_id = arm_entries.get(outcome.key)
                session.store.commit_embedding_outcome(
                    session.experiment_id,
                    record,
                    (
                        {
                            entry_id: (
                                "settled" if outcome.dispatched else "released",
                                record.cost,
                            )
                        }
                        if entry_id is not None
                        else {}
                    ),
                )
            if reservations and ledger is not None:
                # Batches after a failure (or never needed) were not sent.
                ledger.release_reserved(list(reservations.values()))
                session.store.settle_entries(
                    session.experiment_id,
                    {
                        arm_entries[key]: ("released", Decimal("0"))
                        for key in reservations
                        if key in arm_entries
                    },
                )
    return indexes


def _coordinate_tasks(tasks: Iterator[_Task], session: _Session) -> StopReason:
    """Schedule tasks, checking cancellation before claiming each new one."""

    config = session.config
    pending: dict[Future[_TaskOutput], _Claim] = {}
    no_more_tasks = False
    stop: StopReason = None
    batches = session.plan.case_batches if session.plan is not None else None
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        while pending or not no_more_tasks:
            while len(pending) < config.max_workers and not no_more_tasks and stop is None:
                try:
                    task = next(tasks)
                except StopIteration:
                    no_more_tasks = True
                    break
                if session.store.cancel_requested(session.experiment_id):
                    stop = "cancelled"
                    break
                claim = _claim(task, session)
                if claim is None:
                    stop = "budget"
                    break
                future = executor.submit(
                    _evaluate_task,
                    task,
                    config,
                    session.providers,
                    batches,
                    _dispatch_gate(session.journal, session.experiment_id, claim.entry_ids),
                )
                pending[future] = claim

            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(
                completed,
                key=lambda item: (
                    pending[item].task.case.id,
                    pending[item].task.config_id,
                ),
            ):
                claim = pending.pop(future)
                if _settle_task(future, claim, session) and stop is None:
                    stop = "budget"
    return stop


def _claim(task: _Task, session: _Session) -> _Claim | None:
    """Reserve and journal one task's budget; None when the budget is exhausted."""

    if session.ledger is None or session.plan is None:
        return _Claim(task=task, reservations={}, entry_ids={})
    case_calls = session.plan.case_calls(task.retrieval.retriever)
    try:
        reserved = session.ledger.reserve_many(list(case_calls.values()))
    except BudgetExceeded:
        return None
    reservations = dict(zip(case_calls, reserved, strict=True))
    entry_ids = session.store.reserve_entries(
        session.experiment_id,
        task_key=_case_task_key(task.config_id, task.case.id),
        config_id=task.config_id,
        case_id=task.case.id,
        reservations=reservations,
    )
    return _Claim(task=task, reservations=reservations, entry_ids=entry_ids)


def _settle_task(
    future: Future[_TaskOutput], claim: _Claim, session: _Session
) -> bool:
    """Settle one finished task and persist it atomically; True on budget stop."""

    task = claim.task
    config = session.config
    ledger = session.ledger
    pricing = session.pricing
    try:
        output = future.result()
    except _TaskFailure as failure:
        records, charges, budget_stopped = _settle_case_embeddings(
            failure.embeddings, claim.reservations, task, session
        )
        if isinstance(failure.cause, _TaskCancelled):
            chat_charges = _reconcile_cancelled_reservations(
                failure, claim.reservations, config, ledger
            )
            cost = sum((charge.charged for charge in chat_charges.values()), Decimal("0"))
            charges.update(chat_charges)
            session.store.commit_case_outcome(
                session.experiment_id,
                _cancelled_case_result(task, config, failure, cost=cost),
                records,
                _settlements(claim, charges),
            )
            return budget_stopped
        chat_charges = _reconcile_failed_reservations(
            failure, claim.reservations, config, ledger
        )
        charges.update(chat_charges)
        cost = sum((charge.charged for charge in chat_charges.values()), Decimal("0"))
        result = _failed_case_result(
            task,
            config,
            failure,
            cost=cost,
            cost_estimated=any(charge.estimated for charge in chat_charges.values()),
        )
        session.store.commit_case_outcome(
            session.experiment_id, result, records, _settlements(claim, charges)
        )
        return budget_stopped

    result = output.result
    records, charges, budget_stopped = _settle_case_embeddings(
        output.embeddings, claim.reservations, task, session
    )
    if ledger is not None and pricing is not None:
        call_usages = _result_call_usages(result, config)
        chat_phases = [
            phase for phase in (GENERATION_PHASE, JUDGE_PHASE) if phase in claim.reservations
        ]
        phase_costs = [
            calculate_actual_cost(usage, pricing.models[model]) for model, usage in call_usages
        ]
        try:
            actual_cost = ledger.settle_many(
                _chat_reservations(claim.reservations), call_usages
            )
        except BudgetExceeded:
            actual_cost = sum(phase_costs, start=Decimal("0"))
            budget_stopped = True
        for phase, phase_cost in zip(chat_phases, phase_costs, strict=True):
            charges[phase] = _PhaseCharge("settled", phase_cost)
        result = result.model_copy(update={"cost": actual_cost})
    session.store.commit_case_outcome(
        session.experiment_id, result, records, _settlements(claim, charges)
    )
    return budget_stopped


def _settlements(claim: _Claim, charges: dict[str, _PhaseCharge]) -> dict[int, Settlement]:
    return {
        claim.entry_ids[phase]: (charge.state, charge.charged)
        for phase, charge in charges.items()
        if phase in claim.entry_ids
    }


def _dispatch_gate(
    journal: DispatchJournal | None,
    experiment_id: str,
    entry_ids: dict[str, int],
    *,
    check_cancel: bool = True,
) -> DispatchMarker | None:
    """Return the callback every phase passes right before sending a request.

    Once cancellation is observed it raises instead, so no phase of any case
    arm sends another request; otherwise it journals the phase as sent.
    """

    if journal is None:
        return None

    def gate(phase: str) -> None:
        if check_cancel and journal.cancel_requested(experiment_id):
            raise _TaskCancelled(phase)
        entry_id = entry_ids.get(phase)
        if entry_id is not None:
            journal.mark_dispatched(entry_id)

    return gate


def _settle_case_embeddings(
    outcomes: Sequence[_EmbeddingOutcome],
    reservations: dict[str, Decimal],
    task: _Task,
    session: _Session,
) -> tuple[list[EmbeddingCallRecord], dict[str, _PhaseCharge], bool]:
    """Settle each planned per-case embedding batch; release unsent ones."""

    ledger = session.ledger
    records: list[EmbeddingCallRecord] = []
    charges: dict[str, _PhaseCharge] = {}
    budget_stopped = False
    by_key = {outcome.key: outcome for outcome in outcomes}
    for phase, reservation in reservations.items():
        if phase.split("#", maxsplit=1)[0] not in CASE_EMBEDDING_PHASES:
            continue
        outcome = by_key.get(phase)
        if outcome is None:
            if ledger is not None:
                ledger.release_reserved([reservation])
            charges[phase] = _PhaseCharge("released", Decimal("0"))
            continue
        record, exceeded = _settle_embedding(
            outcome,
            reservation=reservation,
            ledger=ledger,
            pricing=session.pricing,
            model=session.config.provider.embedding_model,
            config_id=task.config_id,
            case_id=task.case.id,
        )
        records.append(record)
        charges[phase] = _PhaseCharge(
            "settled" if outcome.dispatched else "released",
            record.cost,
            record.cost_estimated,
        )
        budget_stopped = budget_stopped or exceeded
    return records, charges, budget_stopped


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
            batch_index=outcome.batch_index,
            batch_count=outcome.batch_count,
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


def _observed_embed_batches(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    model: str | None,
    batches: _PhaseBatches,
    mark: DispatchMarker | None = None,
) -> tuple[list[list[float]], list[_EmbeddingOutcome]]:
    """Send each planned batch as its own observed request, in order.

    A failing batch raises with the outcomes of every batch attempted so far,
    so sent batches are settled and later batches are released unsent.
    """

    if sum(len(batch) for batch in batches.ranges) != len(texts):
        raise ValueError(f"{batches.phase} texts do not match the planned batches")
    vectors: list[list[float]] = []
    outcomes: list[_EmbeddingOutcome] = []
    for index, (batch, cap) in enumerate(zip(batches.ranges, batches.caps, strict=True)):
        try:
            batch_vectors, outcome = _observed_embed(
                provider,
                [texts[position] for position in batch],
                model=model,
                phase=batches.phase,
                token_cap=cap,
                mark=mark,
                batch_index=index,
                batch_count=len(batches.ranges),
            )
        except _EmbeddingCallError as error:
            raise _EmbeddingCallError([*outcomes, *error.outcomes], error.cause) from error
        vectors.extend(batch_vectors)
        outcomes.append(outcome)
    return vectors, outcomes


def _observed_embed(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    model: str | None,
    phase: EmbeddingPhase,
    token_cap: int | None,
    mark: DispatchMarker | None = None,
    batch_index: int = 0,
    batch_count: int = 1,
) -> tuple[list[list[float]], _EmbeddingOutcome]:
    """Embed texts within an optional reserved token cap and observe the request.

    ``mark`` journals the batch as sent after the cap check and immediately
    before the provider call, so a refused request is never recorded as sent.
    """

    bound = _embedding_token_bound(texts)
    if token_cap is not None and bound > token_cap:
        cap_error = _EmbeddingCapExceeded(
            f"{phase} input upper bound {bound} exceeds the reserved cap {token_cap}"
        )
        raise _EmbeddingCallError(
            [
                _EmbeddingOutcome(
                    phase=phase,
                    text_count=len(texts),
                    input_token_upper_bound=bound,
                    error=cap_error,
                    dispatched=False,
                    batch_index=batch_index,
                    batch_count=batch_count,
                )
            ],
            cap_error,
        )
    if mark is not None:
        mark(_batch_key(phase, batch_index))
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
            [
                _EmbeddingOutcome(
                    phase=phase,
                    text_count=len(texts),
                    input_token_upper_bound=bound,
                    error=error,
                    dispatched=not not_dispatched,
                    batch_index=batch_index,
                    batch_count=batch_count,
                )
            ],
            error,
        ) from error
    outcome = _EmbeddingOutcome(
        phase=phase,
        text_count=len(texts),
        input_token_upper_bound=bound,
        response=response,
        batch_index=batch_index,
        batch_count=batch_count,
    )
    if len(response.vectors) != len(texts):
        count_error = ValueError("embedding provider returned an unexpected vector count")
        raise _EmbeddingCallError(
            [
                _EmbeddingOutcome(
                    phase=phase,
                    text_count=len(texts),
                    input_token_upper_bound=bound,
                    response=response,
                    error=count_error,
                    batch_index=batch_index,
                    batch_count=batch_count,
                )
            ],
            count_error,
        )
    return response.vectors, outcome


def _embed_phase(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    model: str | None,
    phase: EmbeddingPhase,
    batches: _PhaseBatches | None,
    mark: DispatchMarker | None,
) -> tuple[list[list[float]], list[_EmbeddingOutcome]]:
    """Embed one per-case phase in planned batches, or in one unmetered request."""

    if batches is not None:
        return _observed_embed_batches(
            provider, texts, model=model, batches=batches, mark=mark
        )
    vectors, outcome = _observed_embed(
        provider, texts, model=model, phase=phase, token_cap=None, mark=mark
    )
    return vectors, [outcome]


def _tasks(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    indexes: dict[str, Retriever],
    prompt_engine: PromptEngine,
    skip: set[str] | None = None,
) -> Iterator[_Task]:
    """Yield case arms in a stable order, skipping checkpointed task keys."""

    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        for case in dataset.cases:
            if skip is not None and _case_task_key(config_id, case.id) in skip:
                continue
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
    batches: _CaseBatches | None = None,
    mark: DispatchMarker | None = None,
) -> _TaskOutput:
    embeddings: list[_EmbeddingOutcome] = []
    try:
        if isinstance(task.index, BM25Index):
            # Lexical retrieval is local: no query embedding is sent.
            hits = task.index.search(task.case.question, top_k=task.retrieval.top_k)
        else:
            query_vectors, query_outcomes = _embed_phase(
                providers.embedding,
                [task.case.question],
                model=config.provider.embedding_model,
                phase="embedding_query",
                batches=batches.query if batches is not None else None,
                mark=mark,
            )
            embeddings.extend(query_outcomes)
            hits = task.index.rank(query_vectors[0], top_k=task.retrieval.top_k)
    except _EmbeddingCallError as error:
        embeddings.extend(error.outcomes)
        raise _TaskFailure("retrieval", error.cause, embeddings=embeddings) from error
    except Exception as error:
        raise _TaskFailure("retrieval", error, embeddings=embeddings) from error
    contexts = [f"[{hit.chunk.id}]\n{hit.chunk.text}" for hit in hits]
    try:
        if mark is not None:
            mark(GENERATION_PHASE)
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
        answer_vectors, answer_outcomes = _embed_phase(
            providers.embedding,
            [response.parsed.answer, task.case.reference_answer],
            model=config.provider.embedding_model,
            phase="embedding_answer",
            batches=batches.answer if batches is not None else None,
            mark=mark,
        )
        embeddings.extend(answer_outcomes)
    except _EmbeddingCallError as error:
        embeddings.extend(error.outcomes)
        raise _TaskFailure(
            "metrics", error.cause, response=response, hits=hits, embeddings=embeddings
        ) from error
    except _TaskCancelled as cancelled:
        raise _TaskFailure(
            "metrics", cancelled, response=response, hits=hits, embeddings=embeddings
        ) from cancelled
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
            if mark is not None:
                mark(JUDGE_PHASE)
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
        split=task.case.split,
        review_status=task.case.review.status if task.case.review is not None else None,
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
        split=task.case.split,
        review_status=task.case.review.status if task.case.review is not None else None,
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


def _cancelled_case_result(
    task: _Task, config: ExperimentConfig, failure: _TaskFailure, *, cost: Decimal
) -> CaseResult:
    """Record a case arm stopped by cancellation: settled cost, no metrics."""

    response = failure.response
    return CaseResult(
        case_id=task.case.id,
        question=task.case.question,
        reference_answer=task.case.reference_answer,
        reference_evidence=task.case.reference_evidence,
        category=task.case.category,
        answerability=task.case.answerability,
        difficulty=task.case.difficulty,
        split=task.case.split,
        review_status=task.case.review.status if task.case.review is not None else None,
        config_id=task.config_id,
        model=config.provider.chat_model,
        answer=response.parsed if response is not None else None,
        retrieval_hits=failure.hits,
        usage=response.usage if response is not None else None,
        http_request_count=(
            response.http_request_count if response is not None else 0
        ),
        latency_ms=response.latency_ms if response is not None else 0,
        cost=cost,
        status="cancelled",
        failure_phase=failure.phase,
        error=str(failure.cause),
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
) -> dict[str, _PhaseCharge]:
    """Reconcile generation and judge reservations for one failed case."""

    if ledger is None:
        return {}
    charges: dict[str, _PhaseCharge] = {}

    def release(phase: str) -> None:
        if phase in reservations:
            ledger.release_reserved([reservations[phase]])
            charges[phase] = _PhaseCharge("released", Decimal("0"))

    def charge_cap(phase: str) -> None:
        if phase in reservations:
            charged = ledger.charge_reserved([reservations[phase]])
            charges[phase] = _PhaseCharge("settled", charged, estimated=True)

    def settle_generation() -> None:
        if failure.response is None:
            raise ValueError(f"{failure.phase} failure is missing generation usage")
        if GENERATION_PHASE in reservations:
            charged = ledger.settle_many(
                [reservations[GENERATION_PHASE]],
                [(config.provider.chat_model, failure.response.usage)],
            )
            charges[GENERATION_PHASE] = _PhaseCharge("settled", charged)

    if failure.phase == "retrieval":
        release(GENERATION_PHASE)
        release(JUDGE_PHASE)
    elif failure.phase == "generation":
        charge_cap(GENERATION_PHASE)
        release(JUDGE_PHASE)
    elif failure.phase == "metrics":
        settle_generation()
        release(JUDGE_PHASE)
    else:
        settle_generation()
        charge_cap(JUDGE_PHASE)
    return charges


def _reconcile_cancelled_reservations(
    failure: _TaskFailure,
    reservations: dict[str, Decimal],
    config: ExperimentConfig,
    ledger: BudgetLedger | None,
) -> dict[str, _PhaseCharge]:
    """Settle a returned generation and release the phases cancellation skipped."""

    if ledger is None:
        return {}
    charges: dict[str, _PhaseCharge] = {}
    if GENERATION_PHASE in reservations:
        if failure.response is not None:
            charged = ledger.settle_many(
                [reservations[GENERATION_PHASE]],
                [(config.provider.chat_model, failure.response.usage)],
            )
            charges[GENERATION_PHASE] = _PhaseCharge("settled", charged)
        else:
            ledger.release_reserved([reservations[GENERATION_PHASE]])
            charges[GENERATION_PHASE] = _PhaseCharge("released", Decimal("0"))
    if JUDGE_PHASE in reservations:
        ledger.release_reserved([reservations[JUDGE_PHASE]])
        charges[JUDGE_PHASE] = _PhaseCharge("released", Decimal("0"))
    return charges


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
    unknown_calls: Sequence[LedgerEntry] = (),
) -> dict[str, float]:
    completed = [result for result in results if result.status == "completed"]
    case_by_id = {case.id: case for case in dataset.cases}
    answerable = [
        result
        for result in completed
        if case_by_id[result.case_id].answerability is Answerability.ANSWERABLE
    ]
    cancelled = [result for result in results if result.status == "cancelled"]
    summary: dict[str, float] = {
        "completed_cases": float(len(completed)),
        "failure_count": float(len(results) - len(completed) - len(cancelled)),
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
    if cancelled:
        # Cancelled case arms keep their settled cost but are never scored.
        summary["cancelled_case_count"] = float(len(cancelled))
    if unknown_calls:
        # Calls sent before a crash but never settled are charged at their cap.
        unknown_cost = sum((call.charged for call in unknown_calls), Decimal("0"))
        summary["unknown_outcome_count"] = float(
            len({call.task_key for call in unknown_calls})
        )
        summary["unknown_outcome_cost"] = float(unknown_cost)
        summary["total_cost"] = float(
            sum((result.cost for result in results), Decimal("0"))
            + sum((call.cost for call in embedding_calls), Decimal("0"))
            + unknown_cost
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

    Remote embedding calls are planned per request batch, split by
    ``provider.embedding_batch_size`` and ``embedding_batch_token_limit``:
    the index batches of each dense arm, the query batches of each dense case
    arm, and the answer batches of every case arm. Without an
    ``embedding_provider`` every chunk is counted; with one, only chunks
    whose cache entry matches provider, model, text, and chunking are deducted.
    The local ``fake-hash`` embedding and BM25 retrieval send no requests and
    are not planned.
    """

    return _live_plan(config, dataset, documents, embedding_provider).calls()


def _live_plan(
    config: ExperimentConfig,
    dataset: EvaluationDataset,
    documents: Sequence[Document],
    embedding_provider: EmbeddingProvider | None,
    *,
    remaining: Sequence[tuple[str, str]] | None = None,
) -> _LivePlan:
    """Plan capped calls for the ``remaining`` (config ID, case ID) case arms.

    ``None`` plans every case arm. Indexes are planned for the dense arms
    that still have case arms to run.
    """

    tasks = (
        [
            (_config_id(retrieval), case.id)
            for retrieval in config.retrieval
            for case in dataset.cases
        ]
        if remaining is None
        else list(remaining)
    )
    retrievers = {_config_id(retrieval): retrieval.retriever for retrieval in config.retrieval}
    arms = {config_id for config_id, _ in tasks}
    case_count = len(tasks)
    dense_case_count = sum(
        1 for config_id, _ in tasks if retrievers.get(config_id) == "embedding"
    )
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
            query_embedding=[],
            answer_embedding=[],
            case_batches=None,
        )

    def batch_calls(batches: _PhaseBatches, count: int) -> list[PlannedCall]:
        return [
            PlannedCall(
                model=model,
                input_token_cap=cap * attempts,
                output_token_cap=0,
                count=count,
                phase=_batch_key(batches.phase, index),
                requests_per_case=attempts,
            )
            for index, cap in enumerate(batches.caps)
        ]

    index_embeddings: dict[str, list[PlannedCall]] = {}
    for retrieval in config.retrieval:
        config_id = _config_id(retrieval)
        if retrieval.retriever != "embedding" or config_id not in arms:
            continue
        chunks = _arm_chunks(documents, retrieval)
        if embedding_provider is not None:
            chunks = uncached_chunks(
                chunks,
                embedding_provider,
                model=model,
                cache_path=_embedding_cache_path(config, config_id),
            )
        if chunks:
            index_batches = _PhaseBatches.plan(
                "embedding_index",
                [embedding_text_token_bound(chunk.text) for chunk in chunks],
                config,
            )
            index_embeddings[config_id] = batch_calls(index_batches, 1)
    case_batches = _CaseBatches(
        query=_PhaseBatches.plan(
            "embedding_query",
            [max(embedding_text_token_bound(case.question) for case in dataset.cases)],
            config,
        ),
        answer=_PhaseBatches.plan(
            "embedding_answer",
            [
                GENERATED_ANSWER_EMBEDDING_BYTE_CAP + EMBEDDING_TEXT_TOKEN_ALLOWANCE,
                max(embedding_text_token_bound(case.reference_answer) for case in dataset.cases),
            ],
            config,
        ),
    )
    return _LivePlan(
        generation=generation,
        judge=judge,
        index_embeddings=index_embeddings,
        query_embedding=(
            batch_calls(case_batches.query, dense_case_count) if dense_case_count else []
        ),
        answer_embedding=batch_calls(case_batches.answer, case_count) if case_count else [],
        case_batches=case_batches,
    )


def _batch_key(phase: str, batch_index: int) -> str:
    """Ledger phase of one embedding batch; the first batch keeps the phase name."""

    return phase if batch_index == 0 else f"{phase}#{batch_index}"


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
    documents: Sequence[Document],
    pricing: PricingConfig | None,
    holdout: HoldoutStatus | None = None,
) -> ExperimentIdentity:
    commit_sha, dirty = _git_identity()
    return ExperimentIdentity(
        name=config.name,
        mode=config.mode,
        commit_sha=commit_sha,
        dirty=dirty,
        dataset_hash=dataset.content_hash(),
        dataset_version=dataset.version,
        holdout_freeze_hash=(
            holdout.holdout_hash
            if holdout is not None and holdout.state == "frozen"
            else None
        ),
        prompt_hashes=prompt_engine.hashes(),
        config=config.model_dump(mode="json"),
        random_seed=config.random_seed,
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(),
        corpus_hash=_model_hash(
            [
                {"id": document.id, "title": document.title, "text": document.text}
                for document in documents
            ]
        ),
        prompt_version=PROMPT_VERSION,
        pricing_snapshot=(
            pricing.model_dump(mode="json") if pricing is not None else None
        ),
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
    return canonical_hash(payload)


def _case_task_key(config_id: str, case_id: str) -> str:
    """Stable checkpoint identity of one case arm."""

    return f"case/{config_id}/{case_id}"


def _index_task_key(config_id: str) -> str:
    return f"index/{config_id}"


def _config_id(retrieval: RetrievalConfig) -> str:
    return retrieval.config_id


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int((len(ordered) * percentile) + 0.999999) - 1)
    return ordered[index]
