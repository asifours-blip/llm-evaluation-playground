"""Evaluation dataset models and invariants."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

Difficulty = Literal["easy", "medium", "hard"]


class Answerability(StrEnum):
    """Whether the controlled corpus supports answering a case."""

    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


class EvaluationCase(BaseModel):
    """One question with stable document-level ground truth."""

    id: str
    question: str
    reference_answer: str
    answerability: Answerability
    expected_document_ids: list[str] = Field(default_factory=list)
    reference_evidence: list[str] = Field(default_factory=list)
    category: str
    difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answerability(self) -> Self:
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
        category: str,
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
        category: str,
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

    version: str
    name: str
    description: str = ""
    cases: list[EvaluationCase]

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("dataset case IDs must be unique")
        return self


class ProviderConfig(BaseModel):
    """Connection details for an OpenAI-compatible provider."""

    name: str
    base_url: AnyHttpUrl
    api_key_env: str
    chat_model: str
    embedding_model: str
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class RetrievalConfig(BaseModel):
    """One retrieval arm in an experiment matrix."""

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    prompt_variant: Literal["direct", "evidence_first"]

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


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

    def is_stale(self, on_date: date, max_age_days: int = 7) -> bool:
        return (on_date - self.verified_at).days > max_age_days


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
    database_path: Path
    artifact_dir: Path
    random_seed: int = 42
    provider: ProviderConfig
    retrieval: list[RetrievalConfig]
    budget: BudgetConfig
    pricing_path: Path | None = None

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        retrieval_keys = [
            (arm.chunk_size, arm.chunk_overlap, arm.top_k, arm.prompt_variant)
            for arm in self.retrieval
        ]
        if len(retrieval_keys) != len(set(retrieval_keys)):
            raise ValueError("retrieval configurations must be unique")
        if self.mode == "live" and self.pricing_path is None:
            raise ValueError("live experiments require a pricing_path")
        return self
