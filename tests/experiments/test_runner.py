from collections.abc import Sequence
from datetime import date
from pathlib import Path

import yaml

from rag_quality_lab.domain.models import (
    BudgetConfig,
    EvaluationCase,
    EvaluationDataset,
    ExperimentConfig,
    ExperimentStatus,
    ProviderConfig,
    ProviderResponse,
    RetrievalConfig,
    StructuredAnswer,
    TokenUsage,
)
from rag_quality_lab.experiments.runner import (
    ProviderBundle,
    planned_calls,
    run_experiment,
)
from rag_quality_lab.prompts.engine import PromptEngine
from rag_quality_lab.providers.fake import (
    FakeChatProvider,
    FakeEmbeddingProvider,
    FakeJudgeProvider,
)
from rag_quality_lab.providers.openai_compatible import ProviderError


class HighUsageChatProvider:
    def __init__(self, answers: dict[str, StructuredAnswer]) -> None:
        self.answers = answers

    def answer(
        self,
        question: str,
        contexts: Sequence[str],
        *,
        model: str,
        instructions: str | None = None,
    ) -> ProviderResponse[StructuredAnswer]:
        del contexts, instructions
        return ProviderResponse[StructuredAnswer](
            parsed=self.answers[question],
            usage=TokenUsage(input_tokens=7000, output_tokens=2048),
            model=model,
            latency_ms=1,
        )


class PartiallyFailingChatProvider:
    def __init__(self, answers: dict[str, StructuredAnswer]) -> None:
        self.answers = answers

    def answer(
        self,
        question: str,
        contexts: Sequence[str],
        *,
        model: str,
        instructions: str | None = None,
    ) -> ProviderResponse[StructuredAnswer]:
        del contexts, instructions
        if "weather" in question:
            raise ProviderError("simulated provider failure")
        return ProviderResponse[StructuredAnswer](
            parsed=self.answers[question],
            usage=TokenUsage(input_tokens=20, output_tokens=10),
            model=model,
        )


def scripted_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        version="1.0.0",
        name="scripted",
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


def scripted_answers() -> dict[str, StructuredAnswer]:
    return {
        "What does RAG retrieve?": StructuredAnswer(
            answer="RAG retrieves evidence.",
            citations=["doc-01"],
            abstained=False,
        ),
        "What is tomorrow's weather?": StructuredAnswer(
            answer="The supplied corpus does not contain that information.",
            citations=[],
            abstained=True,
        ),
    }


def write_corpus(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "doc-01-rag.md").write_text(
        "# RAG\n\nID: doc-01\n\nRAG retrieves evidence.",
        encoding="utf-8",
    )


def experiment_config(tmp_path: Path, *, mode: str = "mock") -> ExperimentConfig:
    corpus_path = tmp_path / "knowledge_base"
    write_corpus(corpus_path)
    return ExperimentConfig(
        name=f"{mode}-test",
        mode=mode,
        dataset_path=tmp_path / "dataset.json",
        knowledge_base_path=corpus_path,
        database_path=tmp_path / "experiments.sqlite3",
        artifact_dir=tmp_path / "artifacts",
        max_workers=1,
        provider=ProviderConfig(
            name="fake",
            base_url="https://example.com/v1",
            api_key_env="FAKE_API_KEY",
            chat_model="fake-model",
            embedding_model="fake-embedding",
        ),
        retrieval=[
            RetrievalConfig(
                chunk_size=200,
                chunk_overlap=20,
                top_k=1,
                prompt_variant="direct",
            )
        ],
        budget=BudgetConfig(hard_limit=20),
    )


def test_prompt_hashes_are_stable_and_variant_specific() -> None:
    engine = PromptEngine()

    assert engine.prompt_hash("direct") == engine.prompt_hash("direct")
    assert engine.prompt_hash("direct") != engine.prompt_hash("evidence_first")
    assert "insufficient" in engine.instructions("direct")


def test_offline_runner_persists_separate_retrieval_and_generation_metrics(
    tmp_path: Path,
) -> None:
    config = experiment_config(tmp_path)
    bundle = ProviderBundle(
        embedding=FakeEmbeddingProvider(dimensions=32),
        chat=FakeChatProvider(scripted_answers()),
    )

    result = run_experiment(config, bundle, scripted_dataset())

    assert result.status is ExperimentStatus.COMPLETED
    assert len(result.case_results) == 2
    assert result.case_results[0].metrics["retrieval_recall_at_k"] == 1.0
    assert "answer_f1" in result.case_results[0].metrics
    assert result.summary["false_answer_rate"] == 0.0
    assert result.identity.mode == "mock"


def test_runner_persists_judge_score_usage_and_summary(tmp_path: Path) -> None:
    config = experiment_config(tmp_path)
    config = config.model_copy(
        update={
            "provider": config.provider.model_copy(
                update={"judge_model": "fake-judge"}
            )
        }
    )
    bundle = ProviderBundle(
        embedding=FakeEmbeddingProvider(dimensions=32),
        chat=FakeChatProvider(scripted_answers()),
        judge=FakeJudgeProvider(),
    )

    result = run_experiment(config, bundle, scripted_dataset())

    assert all(case.judge is not None for case in result.case_results)
    assert all(case.judge_usage is not None for case in result.case_results)
    assert {case.metrics["judge_score"] for case in result.case_results} == {2.0, 5.0}
    assert result.summary["judge_mean_score"] == 3.5
    assert result.summary["judge_pass_rate"] == 0.5


def test_runner_preflight_budget_exceeded_schedules_no_paid_cases(
    tmp_path: Path,
) -> None:
    config = experiment_config(tmp_path)
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        yaml.safe_dump(
            {
                "provider": "fake",
                "currency": "CNY",
                "verified_at": date.today().isoformat(),
                "source_url": "https://example.com/pricing",
                "models": {
                    "fake-model": {
                        "input_cache_miss": 1000,
                        "output": 1000,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = config.model_copy(
        update={
            "mode": "live",
            "budget": BudgetConfig(hard_limit=30),
            "pricing_path": pricing_path,
        }
    )
    bundle = ProviderBundle(
        embedding=FakeEmbeddingProvider(dimensions=32),
        chat=HighUsageChatProvider(scripted_answers()),
    )

    result = run_experiment(config, bundle, scripted_dataset())

    assert result.status is ExperimentStatus.BUDGET_EXCEEDED
    assert result.case_results == []
    assert result.summary["total_cost"] == 0.0


def test_runner_isolates_provider_failure_and_persists_the_failed_case(
    tmp_path: Path,
) -> None:
    config = experiment_config(tmp_path)
    bundle = ProviderBundle(
        embedding=FakeEmbeddingProvider(dimensions=32),
        chat=PartiallyFailingChatProvider(scripted_answers()),
    )

    result = run_experiment(config, bundle, scripted_dataset())

    assert result.status is ExperimentStatus.COMPLETED
    assert len(result.case_results) == 2
    failed = next(case for case in result.case_results if case.status == "failed")
    assert failed.case_id == "rag-002"
    assert failed.failure_phase == "generation"
    assert failed.error == "simulated provider failure"
    assert result.summary["completed_cases"] == 1.0
    assert result.summary["failure_count"] == 1.0


def test_live_runner_preflights_and_settles_generation_and_judge_costs(
    tmp_path: Path,
) -> None:
    config = experiment_config(tmp_path)
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        yaml.safe_dump(
            {
                "provider": "fake",
                "currency": "CNY",
                "verified_at": date.today().isoformat(),
                "source_url": "https://example.com/pricing",
                "models": {
                    "fake-model": {"input_cache_miss": 1, "output": 1},
                    "fake-judge": {"input_cache_miss": 1, "output": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    config = config.model_copy(
        update={
            "mode": "live",
            "pricing_path": pricing_path,
            "provider": config.provider.model_copy(
                update={"judge_model": "fake-judge"}
            ),
        }
    )
    bundle = ProviderBundle(
        embedding=FakeEmbeddingProvider(dimensions=32),
        chat=HighUsageChatProvider(scripted_answers()),
        judge=FakeJudgeProvider(),
    )

    result = run_experiment(config, bundle, scripted_dataset())

    assert len(planned_calls(config, 2)) == 2
    assert result.status is ExperimentStatus.COMPLETED
    expected_cost = sum(
        case.usage.total_tokens + case.judge_usage.total_tokens
        for case in result.case_results
        if case.usage is not None and case.judge_usage is not None
    ) / 1_000_000
    assert result.summary["total_cost"] == expected_cost
