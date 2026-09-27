"""Live embedding calls must stay inside the preflight and budget boundary.

Every test drives the real ``OpenAICompatibleProvider`` through an in-process
HTTP session double, so each physical request is counted without any network
access or real API key.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_quality_lab.domain.models import (
    BudgetConfig,
    EvaluationCase,
    EvaluationDataset,
    ExperimentConfig,
    ExperimentRecord,
    ExperimentStatus,
    ProviderConfig,
    RetrievalConfig,
)
from rag_quality_lab.experiments.runner import ProviderBundle, run_experiment
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.providers.fake import FakeEmbeddingProvider
from rag_quality_lab.providers.openai_compatible import (
    JUDGE_OUTPUT_TOKEN_CAP,
    OpenAICompatibleProvider,
    ProviderError,
)

API_KEY_ENV = "RAG_QUALITY_TEST_ONLY_KEY"
EMBEDDING_MODEL = "remote-embedding"
CHAT_MODEL = "remote-chat"
JUDGE_MODEL = "remote-judge"
CORPUS_TEXT = "# RAG\n\nID: doc-01\n\nRAG retrieves evidence."


@dataclass
class LocalResponse:
    status_code: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.payload


class LocalProviderSession:
    """In-process OpenAI-compatible endpoint that records every request."""

    def __init__(
        self,
        *,
        embedding_status: int = 200,
        answer_text: str = "RAG retrieves evidence.",
    ) -> None:
        self.embedding_status = embedding_status
        self.answer_text = answer_text
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.vectors = FakeEmbeddingProvider(dimensions=32)

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> LocalResponse:
        del headers, timeout
        path = url.split("/v1", maxsplit=1)[1]
        self.requests.append((path, json))
        if path == "/embeddings":
            return self._embedding_response(json)
        if path == "/chat/completions":
            return self._chat_response(json)
        return LocalResponse(404, {}, text=f"unknown path {path}")

    def count(self, path: str) -> int:
        return sum(1 for request_path, _ in self.requests if request_path == path)

    def _embedding_response(self, body: dict[str, Any]) -> LocalResponse:
        if self.embedding_status != 200:
            return LocalResponse(
                self.embedding_status, {}, text="embedding backend unavailable"
            )
        texts = list(body["input"])
        prompt_tokens = sum(max(1, len(text.encode("utf-8")) // 4) for text in texts)
        return LocalResponse(
            200,
            {
                "data": [
                    {"index": index, "embedding": vector}
                    for index, vector in enumerate(self.vectors.embed(texts))
                ],
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            },
        )

    def _chat_response(self, body: dict[str, Any]) -> LocalResponse:
        if body["max_tokens"] == JUDGE_OUTPUT_TOKEN_CAP:
            content: dict[str, Any] = {"score": 4, "passed": True, "reason": "grounded"}
        else:
            content = {"answer": self.answer_text, "citations": [], "abstained": False}
        return LocalResponse(
            200,
            {
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )


def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        version="1.0.0",
        name="embedding-boundary",
        cases=[
            EvaluationCase.answerable(
                id="rag-001",
                question="What does RAG retrieve?",
                reference_answer="RAG retrieves evidence.",
                expected_document_ids=["doc-01"],
                reference_evidence=["RAG retrieves evidence."],
                category="retrieval",
                difficulty="easy",
            ),
            EvaluationCase.unanswerable(
                id="rag-002",
                question="What is tomorrow's weather?",
                reference_answer="The corpus has no weather forecast.",
                category="abstention",
                difficulty="easy",
            ),
        ],
    )


def write_pricing(path: Path, prices: dict[str, dict[str, float]]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "provider": "local-test",
                "currency": "CNY",
                "verified_at": date.today().isoformat(),
                "source_url": "https://example.com/pricing",
                "models": prices,
            }
        ),
        encoding="utf-8",
    )
    return path


def full_prices(rate: float = 1) -> dict[str, dict[str, float]]:
    return {
        CHAT_MODEL: {"input_cache_miss": rate, "output": rate},
        JUDGE_MODEL: {"input_cache_miss": rate, "output": rate},
        EMBEDDING_MODEL: {"input_cache_miss": rate, "output": 0},
    }


def live_config(
    tmp_path: Path,
    *,
    prices: dict[str, dict[str, float]] | None = None,
    embedding_model: str = EMBEDDING_MODEL,
    artifact_dir: Path | None = None,
) -> ExperimentConfig:
    corpus = tmp_path / "knowledge_base"
    if not corpus.exists():
        corpus.mkdir(parents=True)
        (corpus / "doc-01-rag.md").write_text(CORPUS_TEXT, encoding="utf-8")
    pricing_path = write_pricing(tmp_path / "pricing.yaml", prices or full_prices())
    return ExperimentConfig(
        name="live-embedding-boundary",
        mode="live",
        dataset_path=tmp_path / "dataset.json",
        knowledge_base_path=corpus,
        database_path=tmp_path / "experiments.sqlite3",
        artifact_dir=artifact_dir or tmp_path / "artifacts",
        max_workers=1,
        provider=ProviderConfig(
            name="local-test",
            base_url="https://provider.test/v1",
            api_key_env=API_KEY_ENV,
            chat_model=CHAT_MODEL,
            embedding_model=embedding_model,
            judge_model=JUDGE_MODEL,
            max_retries=1,
        ),
        retrieval=[
            RetrievalConfig(
                chunk_size=200, chunk_overlap=20, top_k=1, prompt_variant="direct"
            )
        ],
        budget=BudgetConfig(hard_limit=20),
        pricing_path=pricing_path,
    )


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> LocalProviderSession:
    monkeypatch.setenv(API_KEY_ENV, "test-only-not-a-real-key")
    return LocalProviderSession()


def remote_bundle(session: LocalProviderSession, *, max_retries: int = 1) -> ProviderBundle:
    provider = OpenAICompatibleProvider(
        base_url="https://provider.test/v1",
        api_key_env=API_KEY_ENV,
        max_retries=max_retries,
        session=session,
        sleeper=lambda _: None,
        jitter=lambda: 0.0,
    )
    return ProviderBundle(embedding=provider, chat=provider, judge=provider)


def stored_experiments(database_path: Path) -> list[ExperimentRecord]:
    with ExperimentStore(database_path) as store:
        rows = store.connection.execute("SELECT id FROM experiments").fetchall()
        return [store.get_experiment(str(row[0])) for row in rows]


def test_budget_rejection_sends_no_embedding_generation_or_judge_requests(
    tmp_path: Path, session: LocalProviderSession
) -> None:
    config = live_config(tmp_path, prices=full_prices(rate=1000))

    result = run_experiment(config, remote_bundle(session), dataset())

    assert session.requests == []
    assert result.status is ExperimentStatus.BUDGET_EXCEEDED
    assert result.case_results == []
    assert not (config.artifact_dir / "chunk200-overlap20-top1-direct.embeddings.json").exists()


def test_missing_embedding_price_fails_explicitly_before_any_request(
    tmp_path: Path, session: LocalProviderSession
) -> None:
    prices = full_prices()
    del prices[EMBEDDING_MODEL]
    config = live_config(tmp_path, prices=prices)

    with pytest.raises(ValueError, match=EMBEDDING_MODEL):
        run_experiment(config, remote_bundle(session), dataset())

    assert session.requests == []
    experiments = stored_experiments(config.database_path)
    assert [record.status for record in experiments] == [ExperimentStatus.FAILED]


def test_local_embedding_model_name_with_remote_provider_is_rejected(
    tmp_path: Path, session: LocalProviderSession
) -> None:
    config = live_config(tmp_path, embedding_model="fake-hash-32")

    with pytest.raises(ValueError, match="local embedding"):
        run_experiment(config, remote_bundle(session), dataset())

    assert session.requests == []


def test_missing_judge_provider_is_rejected_before_any_request(
    tmp_path: Path, session: LocalProviderSession
) -> None:
    config = live_config(tmp_path)
    bundle = remote_bundle(session)

    with pytest.raises(ValueError, match="judge provider"):
        run_experiment(
            config,
            ProviderBundle(embedding=bundle.embedding, chat=bundle.chat),
            dataset(),
        )

    assert session.requests == []


def test_live_run_plans_settles_and_records_each_embedding_phase(
    tmp_path: Path, session: LocalProviderSession
) -> None:
    from rag_quality_lab.experiments.runner import planned_calls
    from rag_quality_lab.providers.openai_compatible import (
        EMBEDDING_TEXT_TOKEN_ALLOWANCE,
    )
    from rag_quality_lab.retrieval.index import load_documents

    config = live_config(tmp_path)
    cases = dataset()
    bundle = remote_bundle(session)
    plan = planned_calls(
        config,
        cases,
        load_documents(config.knowledge_base_path),
        embedding_provider=bundle.embedding,
    )

    by_phase = {call.phase: call for call in plan}
    assert list(by_phase) == [
        "generation_with_repair",
        "judge_with_repair",
        "embedding_index",
        "embedding_query",
        "embedding_answer",
    ]
    attempts = config.provider.max_retries + 1
    index_call = by_phase["embedding_index"]
    assert index_call.model == EMBEDDING_MODEL
    assert index_call.count == 1
    assert index_call.requests_per_case == attempts
    assert index_call.input_token_cap == (
        len(CORPUS_TEXT.encode("utf-8")) + EMBEDDING_TEXT_TOKEN_ALLOWANCE
    ) * attempts
    longest_question = max(len(case.question.encode("utf-8")) for case in cases.cases)
    assert by_phase["embedding_query"].count == len(cases.cases)
    assert by_phase["embedding_query"].input_token_cap == (
        longest_question + EMBEDDING_TEXT_TOKEN_ALLOWANCE
    ) * attempts
    assert by_phase["embedding_answer"].count == len(cases.cases)
    assert all(
        by_phase[phase].output_token_cap == 0
        for phase in ("embedding_index", "embedding_query", "embedding_answer")
    )

    result = run_experiment(config, bundle, cases)

    assert result.status is ExperimentStatus.COMPLETED
    calls = result.embedding_calls
    assert [call.phase for call in calls].count("embedding_index") == 1
    assert [call.phase for call in calls].count("embedding_query") == 2
    assert [call.phase for call in calls].count("embedding_answer") == 2
    assert {call.case_id for call in calls if call.phase != "embedding_index"} == {
        "rag-001",
        "rag-002",
    }
    assert all(call.status == "completed" for call in calls)
    assert all(call.http_request_count == 1 for call in calls)
    assert session.count("/embeddings") == sum(
        call.http_request_count or 0 for call in calls
    )
    for call in calls:
        assert call.usage is not None
        assert call.cost_estimated is False
        assert call.cost == Decimal(call.usage.input_tokens) / Decimal(1_000_000)
    embedding_cost = sum((call.cost for call in calls), Decimal("0"))
    case_cost = sum((case.cost for case in result.case_results), Decimal("0"))
    assert embedding_cost > 0
    assert result.summary["embedding_cost"] == float(embedding_cost)
    assert result.summary["total_cost"] == float(case_cost + embedding_cost)


def test_cached_index_is_deducted_only_for_the_same_embedding_model(
    tmp_path: Path, session: LocalProviderSession
) -> None:
    from rag_quality_lab.experiments.runner import planned_calls
    from rag_quality_lab.retrieval.index import load_documents

    config = live_config(tmp_path)
    documents = load_documents(config.knowledge_base_path)
    run_experiment(config, remote_bundle(session), dataset())
    first_run_index_requests = [
        body for path, body in session.requests if path == "/embeddings"
    ]
    assert CORPUS_TEXT in first_run_index_requests[0]["input"]

    cached_plan = planned_calls(
        config, dataset(), documents, embedding_provider=remote_bundle(session).embedding
    )
    assert "embedding_index" not in {call.phase for call in cached_plan}
    uncached_plan = planned_calls(config, dataset(), documents)
    assert "embedding_index" in {call.phase for call in uncached_plan}

    other_model = config.model_copy(
        update={
            "provider": config.provider.model_copy(
                update={"embedding_model": "other-embedding"}
            )
        }
    )
    other_plan = planned_calls(
        other_model,
        dataset(),
        documents,
        embedding_provider=remote_bundle(session).embedding,
    )
    assert "embedding_index" in {call.phase for call in other_plan}

    session.requests.clear()
    second = run_experiment(config, remote_bundle(session), dataset())

    assert second.status is ExperimentStatus.COMPLETED
    assert "embedding_index" not in {call.phase for call in second.embedding_calls}
    assert all(
        CORPUS_TEXT not in body["input"]
        for path, body in session.requests
        if path == "/embeddings"
    )


def test_index_embedding_failure_persists_failed_experiment_and_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-only-not-a-real-key")
    session = LocalProviderSession(embedding_status=503)
    config = live_config(tmp_path)

    with pytest.raises(ProviderError):
        run_experiment(config, remote_bundle(session), dataset())

    attempts = config.provider.max_retries + 1
    assert session.count("/embeddings") == attempts
    assert session.count("/chat/completions") == 0
    experiments = stored_experiments(config.database_path)
    assert [record.status for record in experiments] == [ExperimentStatus.FAILED]
    calls = experiments[0].embedding_calls
    assert len(calls) == 1
    index_call = calls[0]
    assert index_call.phase == "embedding_index"
    assert index_call.status == "failed"
    assert index_call.http_request_count == attempts
    assert index_call.cost_estimated is True
    assert index_call.cost > 0


def test_oversized_answer_is_not_embedded_beyond_the_reserved_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-only-not-a-real-key")
    session = LocalProviderSession(answer_text="evidence " * 1000)
    config = live_config(tmp_path)

    result = run_experiment(config, remote_bundle(session), dataset())

    assert all(case.status == "failed" for case in result.case_results)
    assert all(case.failure_phase == "metrics" for case in result.case_results)
    answer_calls = [
        call for call in result.embedding_calls if call.phase == "embedding_answer"
    ]
    assert len(answer_calls) == 2
    assert all(call.http_request_count == 0 for call in answer_calls)
    assert all(call.cost == 0 for call in answer_calls)
    assert all(
        "evidence evidence" not in " ".join(body["input"])
        for path, body in session.requests
        if path == "/embeddings"
    )
