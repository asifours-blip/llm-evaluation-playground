"""Evaluation dataset models and invariants."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

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
