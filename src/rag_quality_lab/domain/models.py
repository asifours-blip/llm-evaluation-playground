"""Evaluation dataset models and invariants."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeVar

from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

Difficulty = Literal["easy", "medium", "hard"]
CaseCategory = Literal[
    "abstention",
    "abstention_cost",
    "chunking",
    "chunking_embeddings",
    "cost",
    "deployment",
    "deployment_pipeline",
    "embeddings",
    "embeddings_retrieval",
    "evaluation",
    "evaluation_abstention",
    "failures",
    "failures_deployment",
    "ingestion",
    "ingestion_chunking",
    "latency",
    "latency_failures",
    "out_of_scope",
    "pipeline",
    "prompting",
    "prompting_evaluation",
    "rag_overview",
    "reranking",
    "reranking_prompting",
    "retrieval",
    "retrieval_reranking",
    "unsupported_detail",
]
ResponseT = TypeVar("ResponseT")
DatasetSplit = Literal["dev", "holdout"]
ReviewStatus = Literal["unreviewed", "approved", "rejected"]
RetrieverKind = Literal["embedding", "bm25"]
# Label shown wherever a dataset field was never annotated.
UNLABELED = "未标注"


def canonical_hash(payload: object) -> str:
    """SHA-256 of canonical (sorted, compact, UTF-8) JSON."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class Answerability(StrEnum):
    """Whether the controlled corpus supports answering a case."""

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


class CaseReview(BaseModel):
    """Human review state of one evaluation case."""

    status: ReviewStatus = "unreviewed"
    reviewer: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.status != "unreviewed" and (
            self.reviewer is None or self.reviewed_at is None
        ):
            raise ValueError(f"a {self.status} review requires reviewer and reviewed_at")
        return self


class EvaluationCase(BaseModel):
    """One question with stable document-level ground truth.

    ``difficulty``, ``split``, and ``review`` are optional so datasets written
    before they existed still load; an absent value is reported as unlabeled.
    """

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    answerability: Answerability
    expected_document_ids: list[str] = Field(default_factory=list)
    reference_evidence: list[str] = Field(default_factory=list)
    category: CaseCategory
    difficulty: Difficulty | None = None
    tags: list[str] = Field(default_factory=list)
    split: DatasetSplit | None = None
    review: CaseReview | None = None

    @property
    def is_unanswerable(self) -> bool:
        return self.answerability is Answerability.UNANSWERABLE

    def label_values(self) -> dict[str, str]:
        """Label values for reporting, with absent labels shown as unlabeled."""

        return {
            "difficulty": self.difficulty or UNLABELED,
            "answerability": self.answerability.value,
            "review_status": self.review.status if self.review is not None else UNLABELED,
            "split": self.split or UNLABELED,
        }

    @model_validator(mode="after")
    def validate_answerability(self) -> Self:
        if any(not value.strip() for value in self.expected_document_ids):
            raise ValueError("expected document IDs must be non-empty")
        if any(not value.strip() for value in self.reference_evidence):
            raise ValueError("reference evidence must be non-empty")
        has_documents = bool(self.expected_document_ids)
        has_evidence = bool(self.reference_evidence)
        if self.answerability is Answerability.ANSWERABLE and not (
            has_documents and has_evidence
        ):
            raise ValueError("answerable cases require expected documents and evidence")
        if self.answerability is Answerability.UNANSWERABLE and (
            has_documents or has_evidence
        ):
            raise ValueError("unanswerable cases cannot contain ground-truth evidence")
        return self

    @classmethod
    def answerable(
        cls,
        *,
        id: str,
        question: str,
        reference_answer: str,
        expected_document_ids: list[str],
        reference_evidence: list[str],
        category: CaseCategory,
        difficulty: Difficulty,
        tags: list[str] | None = None,
    ) -> EvaluationCase:
        return cls(
            id=id,
            question=question,
            reference_answer=reference_answer,
            answerability=Answerability.ANSWERABLE,
            expected_document_ids=expected_document_ids,
            reference_evidence=reference_evidence,
            category=category,
            difficulty=difficulty,
            tags=tags or [],
        )

    @classmethod
    def unanswerable(
        cls,
        *,
        id: str,
        question: str,
        reference_answer: str,
        category: CaseCategory,
        difficulty: Difficulty,
        tags: list[str] | None = None,
    ) -> EvaluationCase:
        return cls(
            id=id,
            question=question,
            reference_answer=reference_answer,
            answerability=Answerability.UNANSWERABLE,
            category=category,
            difficulty=difficulty,
            tags=tags or [],
        )


class EvaluationDataset(BaseModel):
    """A versioned collection of evaluation cases."""

    version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("dataset case IDs must be unique")
        return self

    def content_hash(self) -> str:
        """Hash every field of the dataset, including version and labels.

        Unset optional labels are omitted, so a dataset written before labels
        existed keeps the ``dataset_hash`` it was recorded with.
        """

        return canonical_hash(self.model_dump(mode="json", exclude_none=True))

    def split_case_ids(self, split: DatasetSplit) -> list[str]:
        return sorted(case.id for case in self.cases if case.split == split)

    def holdout_hash(self) -> str:
        """Hash the complete content of every holdout case, ordered by case ID."""

        holdout = sorted(
            (case for case in self.cases if case.split == "holdout"),
            key=lambda case: case.id,
        )
        return canonical_hash(
            [case.model_dump(mode="json", exclude_none=True) for case in holdout]
        )

    def select_splits(self, splits: Sequence[DatasetSplit] | None) -> EvaluationDataset:
        """Return the cases of ``splits``; ``None`` keeps every case."""

        if splits is None:
            return self
        selected = [case for case in self.cases if case.split in splits]
        if not selected:
            raise ValueError(
                f"dataset {self.name} {self.version} has no cases in splits: "
                + ", ".join(splits)
            )
        return self.model_copy(update={"cases": selected})


class ProviderConfig(BaseModel):
    """Connection details for an OpenAI-compatible provider."""

    name: str
    base_url: AnyHttpUrl
    api_key_env: str
    chat_model: str
    embedding_model: str
    judge_model: str | None = None
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    # Remote embedding requests carry at most this many texts and this many
    # upper-bound tokens; None sends each embedding phase in one request.
    embedding_batch_size: int | None = Field(default=None, ge=1)
    embedding_batch_token_limit: int | None = Field(default=None, ge=1)


class RetrievalConfig(BaseModel):
    """One retrieval arm in an experiment matrix.

    ``retriever`` selects dense retrieval with ``provider.embedding_model``
    (the local ``fake-hash`` weak baseline or a remote model) or the local
    BM25 lexical baseline, which sends no provider request.
    """

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    prompt_variant: Literal["direct", "evidence_first"]
    retriever: RetrieverKind = "embedding"

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def config_id(self) -> str:
        """Stable arm identity; embedding arms keep their historical IDs."""

        base = (
            f"chunk{self.chunk_size}-overlap{self.chunk_overlap}-"
            f"top{self.top_k}-{self.prompt_variant}"
        )
        return base if self.retriever == "embedding" else f"{base}-{self.retriever}"


class ModelPrice(BaseModel):
    """Per-million-token model prices."""

    input_cache_hit: Decimal | None = Field(default=None, ge=0)
    input_cache_miss: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)


class PricingConfig(BaseModel):
    """Verified provider pricing used for budget estimation."""

    provider: str
    currency: str
    verified_at: date
    source_url: AnyHttpUrl
    models: dict[str, ModelPrice]
    rate_basis: Literal["standard", "peak", "off_peak"] = "standard"
    notes: list[str] = Field(default_factory=list)

    def is_stale(self, on_date: date, max_age_days: int = 7) -> bool:
        return (on_date - self.verified_at).days > max_age_days

    def require_models(self, models: Iterable[str]) -> None:
        """Reject a plan that uses any model without an explicit price."""

        missing = sorted({model for model in models if model not in self.models})
        if missing:
            raise ValueError(
                "missing price for planned model(s): " + ", ".join(missing)
            )


class BudgetConfig(BaseModel):
    """Hard and preflight spend controls."""

    currency: str = "CNY"
    hard_limit: Decimal = Field(gt=0)
    preflight_fraction: Decimal = Field(default=Decimal("0.90"), gt=0, le=1)
    safety_multiplier: Decimal = Field(default=Decimal("1.25"), ge=1)


class ExperimentConfig(BaseModel):
    """Reproducible inputs for one experiment run."""

    name: str
    mode: Literal["mock", "live"]
    dataset_path: Path
    knowledge_base_path: Path = Path("data/knowledge_base")
    database_path: Path
    artifact_dir: Path
    random_seed: int = 42
    max_workers: int = Field(default=4, ge=1, le=16)
    provider: ProviderConfig
    retrieval: list[RetrievalConfig]
    budget: BudgetConfig
    pricing_path: Path | None = None
    # Dataset splits to evaluate; None evaluates every case.
    splits: list[DatasetSplit] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        retrieval_keys = [arm.config_id for arm in self.retrieval]
        if len(retrieval_keys) != len(set(retrieval_keys)):
            raise ValueError("retrieval configurations must be unique")
        if self.mode == "live" and self.pricing_path is None:
            raise ValueError("live experiments require a pricing_path")
        return self


class TokenUsage(BaseModel):
    """Normalized provider token accounting."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    input_cache_hit_tokens: int = Field(default=0, ge=0)
    input_cache_miss_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_cache_breakdown(self) -> Self:
        accounted_input = self.input_cache_hit_tokens + self.input_cache_miss_tokens
        if accounted_input not in {0, self.input_tokens}:
            raise ValueError("cache hit and miss tokens must sum to input_tokens")
        return self

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StructuredAnswer(BaseModel):
    """Machine-readable response produced by the RAG generator."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    abstained: bool


class JudgeVerdict(BaseModel):
    """Structured score returned by a model judge."""

    score: int = Field(ge=1, le=5)
    passed: bool
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pass_threshold(self) -> Self:
        if self.passed != (self.score >= 4):
            raise ValueError("passed must be true exactly when score is 4 or 5")
        return self


class PairwiseVerdict(BaseModel):
    """Order-specific pairwise preference returned by a model judge."""

    preferred: Literal["A", "B", "tie"]
    reason: str = Field(min_length=1)


class ProviderResponse(BaseModel, Generic[ResponseT]):
    """A parsed provider result plus observable execution metadata."""

    parsed: ResponseT
    usage: TokenUsage
    model: str
    latency_ms: float = Field(default=0, ge=0)
    http_request_count: int | None = Field(default=None, ge=0)
    raw: dict[str, Any] | None = None


class EmbeddingResponse(BaseModel):
    """Embedding vectors plus observable request metadata."""

    vectors: list[list[float]]
    model: str | None = None
    usage: TokenUsage | None = None
    http_request_count: int | None = Field(default=None, ge=0)


class Document(BaseModel):
    """A normalized source document with a stable identity."""

    id: str
    title: str
    text: str
    source_path: str


class Chunk(BaseModel):
    """A deterministic retrieval unit derived from a document."""

    id: str
    document_id: str
    text: str
    start_char: int = Field(default=0, ge=0)
    end_char: int = Field(default=0, ge=0)


class ParameterDifference(BaseModel):
    """One parameter whose value differs between two compared configurations."""

    path: str
    baseline: Any = None
    candidate: Any = None


class RetrievalHit(BaseModel):
    """A chunk paired with its query similarity score."""

    chunk: Chunk
    score: float


class ExperimentStatus(StrEnum):
    """Lifecycle states persisted for an experiment.

    RUNNING may become any other state. INTERRUPTED (the owning process
    stopped without finishing) may become RUNNING again through resume, or
    CANCELLED. INCOMPLETE (every other case arm finished, but some have calls
    with unknown outcomes) may become RUNNING only through a resume that
    re-sends those calls, or CANCELLED. COMPLETED, FAILED, BUDGET_EXCEEDED,
    and CANCELLED are terminal.
    """

    RUNNING = "running"
    INTERRUPTED = "interrupted"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


class ExperimentIdentity(BaseModel):
    """Inputs required to reproduce an experiment."""

    name: str
    mode: Literal["mock", "live"]
    commit_sha: str
    dirty: bool
    dataset_hash: str
    prompt_hashes: dict[str, str]
    config: dict[str, Any]
    random_seed: int
    python_version: str
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    # Frozen resume identity; experiments recorded before resume support
    # leave these unset and cannot be resumed.
    corpus_hash: str | None = None
    prompt_version: str | None = None
    pricing_snapshot: dict[str, Any] | None = None
    # Dataset version and the verified frozen holdout hash; unset before
    # dataset versioning existed or when the holdout was not frozen.
    dataset_version: str | None = None
    holdout_freeze_hash: str | None = None


class CaseResult(BaseModel):
    """Persisted outcome for one case, configuration, and model."""

    case_id: str
    question: str = ""
    reference_answer: str = ""
    reference_evidence: list[str] = Field(default_factory=list)
    category: str = ""
    answerability: Answerability | None = None
    difficulty: Difficulty | None = None
    split: DatasetSplit | None = None
    review_status: ReviewStatus | None = None
    config_id: str
    model: str
    answer: StructuredAnswer | None = None
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    usage: TokenUsage | None = None
    judge: JudgeVerdict | None = None
    judge_model: str | None = None
    judge_usage: TokenUsage | None = None
    http_request_count: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    cost: Decimal = Field(default=Decimal("0"), ge=0)
    cost_estimated: bool = False
    # "cancelled": the case arm stopped at a cancellation before finishing;
    # requests already sent are settled, but it carries no metrics.
    status: Literal["completed", "failed", "skipped", "cancelled"]
    failure_phase: Literal["retrieval", "generation", "metrics", "judge"] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_judge_metadata(self) -> Self:
        judge_fields = (self.judge, self.judge_model, self.judge_usage)
        if any(value is not None for value in judge_fields) and not all(
            value is not None for value in judge_fields
        ):
            raise ValueError(
                "judge, judge_model, and judge_usage must be provided together"
            )
        return self


EmbeddingPhase = Literal["embedding_index", "embedding_query", "embedding_answer"]


class EmbeddingCallRecord(BaseModel):
    """Persisted outcome of one budgeted live embedding request batch."""

    phase: EmbeddingPhase
    config_id: str
    case_id: str | None = None
    batch_index: int = Field(default=0, ge=0)
    batch_count: int = Field(default=1, ge=1)
    model: str
    text_count: int = Field(ge=0)
    input_token_upper_bound: int = Field(ge=0)
    usage: TokenUsage | None = None
    http_request_count: int | None = Field(default=None, ge=0)
    cost: Decimal = Field(default=Decimal("0"), ge=0)
    cost_estimated: bool = False
    status: Literal["completed", "failed"]
    error: str | None = None


LedgerEntryState = Literal["reserved", "dispatched", "settled", "released", "unknown"]


class LedgerEntry(BaseModel):
    """Persisted budget reservation for one provider phase of one task.

    ``reserved`` entries were never sent; ``dispatched`` entries were handed
    to the provider. When the owning process stops before settlement, the
    former are released and the latter become ``unknown`` and are charged at
    their full reservation.
    """

    task_key: str
    config_id: str
    case_id: str | None = None
    phase: str
    attempt: int = Field(ge=1)
    state: LedgerEntryState
    reserved: Decimal = Field(ge=0)
    charged: Decimal = Field(default=Decimal("0"), ge=0)


class RunAttempt(BaseModel):
    """One run or resume of an experiment and the code revision that executed it."""

    attempt: int = Field(ge=1)
    kind: Literal["run", "resume"]
    commit_sha: str | None = None
    dirty: bool | None = None
    started_at: str
    ended_at: str | None = None
    end_status: str | None = None


class ExperimentRecord(BaseModel):
    """Typed experiment identity, lifecycle, and case outcomes."""

    id: str
    identity: ExperimentIdentity
    status: ExperimentStatus
    case_results: list[CaseResult] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)
    embedding_calls: list[EmbeddingCallRecord] = Field(default_factory=list)
    unknown_calls: list[LedgerEntry] = Field(default_factory=list)
    attempts: list[RunAttempt] = Field(default_factory=list)

    def code_versions(self) -> list[str]:
        """Distinct commits that produced the persisted results, in run order."""

        commits: list[str] = []
        for attempt in self.attempts:
            if attempt.commit_sha is not None and attempt.commit_sha not in commits:
                commits.append(attempt.commit_sha)
        return commits

    def mixed_code_versions(self) -> bool:
        return len(self.code_versions()) > 1

    def code_version_warning(self) -> str | None:
        """Explain which attempt ran which commit when results mix code versions."""

        if not self.mixed_code_versions():
            return None
        runs = ", ".join(
            f"attempt {attempt.attempt} ({attempt.kind}) at {attempt.commit_sha}"
            + (" with uncommitted changes" if attempt.dirty else "")
            for attempt in self.attempts
        )
        return f"results come from multiple code versions: {runs}"
