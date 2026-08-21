import math

import pytest

from rag_quality_lab.domain.models import StructuredAnswer
from rag_quality_lab.providers.base import ChatProvider, EmbeddingProvider, JudgeProvider
from rag_quality_lab.providers.fake import (
    FakeChatProvider,
    FakeEmbeddingProvider,
    FakeJudgeProvider,
)


def test_fake_embedding_is_deterministic_and_normalized() -> None:
    provider = FakeEmbeddingProvider(dimensions=32)

    first = provider.embed(["chunk overlap"])[0]
    second = provider.embed(["chunk overlap"])[0]

    assert first == second
    assert len(first) == 32
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)
    assert isinstance(provider, EmbeddingProvider)


def test_scripted_chat_returns_structured_answer() -> None:
    provider = FakeChatProvider(
        answers={
            "What is RAG?": StructuredAnswer(
                answer="Retrieval-augmented generation.",
                citations=["doc-01#chunk-000"],
                abstained=False,
            )
        }
    )

    response = provider.answer("What is RAG?", [], model="fake-model")

    assert response.parsed.answer.startswith("Retrieval")
    assert response.usage.total_tokens > 0
    assert response.http_request_count == 0
    assert isinstance(provider, ChatProvider)


def test_scripted_chat_rejects_unscripted_question() -> None:
    provider = FakeChatProvider(answers={})

    with pytest.raises(KeyError, match="Unscripted question"):
        provider.answer("Unknown", [], model="fake-model")


def test_fake_judge_scores_exact_reference_deterministically() -> None:
    provider = FakeJudgeProvider()

    response = provider.judge(
        "What is RAG?",
        "Retrieval-augmented generation.",
        "Retrieval-augmented generation.",
        ["RAG retrieves evidence."],
        model="fake-judge",
    )

    assert response.parsed.score == 5
    assert response.parsed.passed
    assert response.usage.total_tokens > 0
    assert response.http_request_count == 0
    assert isinstance(provider, JudgeProvider)
