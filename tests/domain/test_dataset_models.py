import pytest
from pydantic import ValidationError

from rag_quality_lab.domain.models import Answerability, EvaluationCase, EvaluationDataset


def test_answerable_case_requires_documents_and_evidence() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="rag-001",
            question="What is chunk overlap?",
            reference_answer="It repeats boundary text.",
            answerability=Answerability.ANSWERABLE,
            expected_document_ids=[],
            reference_evidence=[],
            category="retrieval",
            difficulty="easy",
        )


def test_unanswerable_case_rejects_ground_truth_evidence() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="rag-002",
            question="Unsupported question",
            reference_answer="The corpus does not contain this information.",
            answerability=Answerability.UNANSWERABLE,
            expected_document_ids=["doc-01"],
            reference_evidence=["unsupported"],
            category="abstention",
            difficulty="medium",
        )


def test_dataset_rejects_duplicate_case_ids() -> None:
    case = EvaluationCase.answerable(
        id="rag-001",
        question="Q",
        reference_answer="A",
        expected_document_ids=["doc-01"],
        reference_evidence=["A"],
        category="retrieval",
        difficulty="easy",
    )

    with pytest.raises(ValidationError):
        EvaluationDataset(version="1.0.0", name="demo", cases=[case, case])


def test_case_constructors_set_answerability() -> None:
    answerable = EvaluationCase.answerable(
        id="rag-001",
        question="Q",
        reference_answer="A",
        expected_document_ids=["doc-01"],
        reference_evidence=["A"],
        category="retrieval",
        difficulty="easy",
    )
    unanswerable = EvaluationCase.unanswerable(
        id="rag-002",
        question="Unsupported question",
        reference_answer="Not in the corpus.",
        category="abstention",
        difficulty="hard",
    )

    assert answerable.answerability is Answerability.ANSWERABLE
    assert unanswerable.answerability is Answerability.UNANSWERABLE
    assert unanswerable.expected_document_ids == []
