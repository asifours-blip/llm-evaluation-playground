"""Embedding requests are split into configured batches and budgeted per batch.

The real ``OpenAICompatibleProvider`` talks to an in-process HTTP double, so
every physical request is counted without network access or a real API key.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from rag_quality_lab.domain.models import ExperimentConfig, ExperimentStatus
from rag_quality_lab.experiments.runner import ProviderBundle, planned_calls, run_experiment
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.providers.openai_compatible import (
    EMBEDDING_TEXT_TOKEN_ALLOWANCE,
    OpenAICompatibleProvider,
    ProviderError,
    split_embedding_batches,
)
from rag_quality_lab.retrieval.index import load_documents
from tests.experiments.test_embedding_budget_boundary import (
    API_KEY_ENV,
    EMBEDDING_MODEL,
    LocalProviderSession,
    LocalResponse,
    dataset,
    live_config,
    stored_experiments,
)

LONG_CORPUS = (
    "# RAG\n\nID: doc-01\n\nRAG retrieves evidence. "
    + " ".join(f"RAG retrieves evidence sentence {index}." for index in range(40))
)


class FailingBatchSession(LocalProviderSession):
    """HTTP double that fails chosen embedding requests (1-based) with a 503."""

    def __init__(self, failing_requests: set[int]) -> None:
        super().__init__()
        self.failing_requests = failing_requests
        self.embedding_requests = 0

    def _embedding_response(self, body: dict[str, Any]) -> LocalResponse:
        self.embedding_requests += 1
        if self.embedding_requests in self.failing_requests:
            return LocalResponse(503, {}, text="embedding backend unavailable")
        return super()._embedding_response(body)


def batched_config(
    tmp_path: Path,
    *,
    batch_size: int | None,
    token_limit: int | None = None,
) -> ExperimentConfig:
    config = live_config(tmp_path)
    (config.knowledge_base_path / "doc-01-rag.md").write_text(LONG_CORPUS, encoding="utf-8")
    return config.model_copy(
        update={
            "provider": config.provider.model_copy(
                update={
                    "max_retries": 0,
                    "embedding_batch_size": batch_size,
                    "embedding_batch_token_limit": token_limit,
                }
            )
        }
    )


def bundle(session: LocalProviderSession, config: ExperimentConfig) -> ProviderBundle:
    provider = OpenAICompatibleProvider(
        base_url="https://provider.test/v1",
        api_key_env=API_KEY_ENV,
        max_retries=config.provider.max_retries,
        session=session,
        sleeper=lambda _: None,
        jitter=lambda: 0.0,
        embedding_batch_size=config.provider.embedding_batch_size,
        embedding_batch_token_limit=config.provider.embedding_batch_token_limit,
    )
    return ProviderBundle(embedding=provider, chat=provider, judge=provider)


def planned_embedding_requests(config: ExperimentConfig) -> int:
    plan = planned_calls(config, dataset(), load_documents(config.knowledge_base_path))
    return sum(
        call.count * call.requests_per_case
        for call in plan
        if call.phase.startswith("embedding")
    )


def embedding_ledger(config: ExperimentConfig, experiment_id: str) -> list[Any]:
    with ExperimentStore(config.database_path) as store:
        return [
            entry
            for entry in store.ledger_entries(experiment_id)
            if entry.phase.startswith("embedding")
        ]


@pytest.fixture(autouse=True)
def test_only_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-only-not-a-real-key")


def test_provider_splits_one_embed_call_into_configured_batches() -> None:
    session = LocalProviderSession()
    provider = OpenAICompatibleProvider(
        base_url="https://provider.test/v1",
        api_key_env=API_KEY_ENV,
        session=session,
        embedding_batch_size=2,
    )
    texts = [f"text {index}" for index in range(5)]

    response = provider.embed_with_metadata(texts, model=EMBEDDING_MODEL)

    assert [len(body["input"]) for _, body in session.requests] == [2, 2, 1]
    assert [body["input"] for _, body in session.requests][2] == ["text 4"]
    assert response.vectors == session.vectors.embed(texts)
    assert response.http_request_count == 3
    assert response.usage is not None
    assert response.usage.input_tokens == sum(
        max(1, len(text.encode("utf-8")) // 4) for text in texts
    )


def test_token_limit_splits_batches_and_rejects_an_unsplittable_text() -> None:
    bound = len(b"abcd") + EMBEDDING_TEXT_TOKEN_ALLOWANCE

    assert split_embedding_batches(
        [bound] * 5, max_inputs=None, max_tokens=bound * 2
    ) == [range(0, 2), range(2, 4), range(4, 5)]
    assert split_embedding_batches([bound] * 3, max_inputs=None, max_tokens=None) == [
        range(0, 3)
    ]
    with pytest.raises(ValueError, match="token limit"):
        split_embedding_batches([bound * 3], max_inputs=None, max_tokens=bound * 2)

    session = LocalProviderSession()
    provider = OpenAICompatibleProvider(
        base_url="https://provider.test/v1",
        api_key_env=API_KEY_ENV,
        session=session,
        embedding_batch_token_limit=bound,
    )
    with pytest.raises(ValueError, match="token limit"):
        provider.embed_with_metadata(["abcd", "abcde"], model=EMBEDDING_MODEL)
    assert session.requests == []


@pytest.mark.parametrize(
    ("batch_size", "token_limit"),
    [(None, None), (2, None), (1, None), (None, 4110)],
)
def test_planned_embedding_requests_equal_requests_received(
    tmp_path: Path, batch_size: int | None, token_limit: int | None
) -> None:
    config = batched_config(tmp_path, batch_size=batch_size, token_limit=token_limit)
    session = LocalProviderSession()
    planned = planned_embedding_requests(config)

    record = run_experiment(config, bundle(session, config), dataset())

    assert record.status is ExperimentStatus.COMPLETED
    assert session.count("/embeddings") == planned
    assert len(record.embedding_calls) == planned
    assert all(call.http_request_count == 1 for call in record.embedding_calls)
    for _, body in session.requests:
        if "input" not in body:
            continue
        if batch_size is not None:
            assert len(body["input"]) <= batch_size
        if token_limit is not None:
            assert sum(
                len(text.encode("utf-8")) + EMBEDDING_TEXT_TOKEN_ALLOWANCE
                for text in body["input"]
            ) <= token_limit
    ledger = embedding_ledger(config, record.id)
    assert len(ledger) == planned
    assert {entry.state for entry in ledger} == {"settled"}
    if batch_size == 2:
        index_batches = [
            call for call in record.embedding_calls if call.phase == "embedding_index"
        ]
        assert len(index_batches) > 1
        assert [call.batch_index for call in index_batches] == list(
            range(len(index_batches))
        )
        assert {call.batch_count for call in index_batches} == {len(index_batches)}


def test_failed_index_batch_charges_sent_batches_and_releases_unsent_ones(
    tmp_path: Path,
) -> None:
    config = batched_config(tmp_path, batch_size=2)
    session = FailingBatchSession(failing_requests={2})
    index_batches = sum(
        1
        for call in planned_calls(config, dataset(), load_documents(config.knowledge_base_path))
        if call.phase.startswith("embedding_index")
    )
    assert index_batches >= 3

    with pytest.raises(ProviderError):
        run_experiment(config, bundle(session, config), dataset())

    assert session.count("/embeddings") == 2
    assert session.count("/chat/completions") == 0
    [record] = stored_experiments(config.database_path)
    assert record.status is ExperimentStatus.FAILED
    calls = sorted(record.embedding_calls, key=lambda call: call.batch_index)
    first, failed = calls[0], calls[1]
    # The successful batch is billed from its reported usage.
    assert first.status == "completed"
    assert first.usage is not None
    assert first.cost == Decimal(first.usage.input_tokens) / Decimal(1_000_000)
    assert first.cost_estimated is False
    # The failed batch was sent without usage, so it is charged at its cap.
    assert failed.status == "failed"
    assert failed.http_request_count == 1
    assert failed.cost_estimated is True
    ledger = embedding_ledger(config, record.id)
    states = [entry.state for entry in ledger]
    assert states.count("settled") == 2
    assert states.count("released") == index_batches - 2
    failed_entry = ledger[1]
    assert failed_entry.charged == failed_entry.reserved == failed.cost
    assert ledger[0].charged == first.cost
    assert all(entry.charged == 0 for entry in ledger if entry.state == "released")


def test_failed_answer_batch_fails_the_case_and_bills_each_batch_by_rule(
    tmp_path: Path,
) -> None:
    config = batched_config(tmp_path, batch_size=1)
    documents = load_documents(config.knowledge_base_path)
    index_requests = sum(
        1 for call in planned_calls(config, dataset(), documents)
        if call.phase.startswith("embedding_index")
    )
    # Request order: index batches, then query, answer#0, answer#1 of case 1.
    failing = index_requests + 3
    session = FailingBatchSession(failing_requests={failing})

    record = run_experiment(config, bundle(session, config), dataset())

    assert session.count("/embeddings") == planned_embedding_requests(config)
    first_case = next(case for case in record.case_results if case.case_id == "rag-001")
    assert first_case.status == "failed"
    assert first_case.failure_phase == "metrics"
    answers = [
        call
        for call in record.embedding_calls
        if call.phase == "embedding_answer" and call.case_id == "rag-001"
    ]
    assert [(call.batch_index, call.status) for call in answers] == [
        (0, "completed"),
        (1, "failed"),
    ]
    assert answers[0].cost_estimated is False
    assert answers[1].cost_estimated is True
    case_entries = [
        entry
        for entry in embedding_ledger(config, record.id)
        if entry.case_id == "rag-001" and entry.phase.startswith("embedding_answer")
    ]
    assert [entry.state for entry in case_entries] == ["settled", "settled"]
    assert case_entries[1].charged == case_entries[1].reserved
    other = next(case for case in record.case_results if case.case_id == "rag-002")
    assert other.status == "completed"
