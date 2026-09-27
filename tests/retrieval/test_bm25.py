"""BM25 lexical baseline: hand-computed ranking, determinism, and configuration."""

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_quality_lab.domain.models import (
    BudgetConfig,
    Chunk,
    EvaluationCase,
    EvaluationDataset,
    ExperimentConfig,
    ExperimentStatus,
    ProviderConfig,
    RetrievalConfig,
    StructuredAnswer,
)
from rag_quality_lab.experiments.runner import ProviderBundle, planned_calls, run_experiment
from rag_quality_lab.providers.fake import FakeChatProvider, FakeEmbeddingProvider
from rag_quality_lab.retrieval.bm25 import BM25_B, BM25_K1, BM25Index, tokenize
from rag_quality_lab.retrieval.index import load_documents


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, document_id=chunk_id.split("#")[0], text=text)


def hand_bm25(tf: int, doc_length: int, average_length: float, df: int, n: int) -> float:
    """Okapi BM25 term weight written out independently of the implementation."""

    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    norm = BM25_K1 * (1 - BM25_B + BM25_B * doc_length / average_length)
    return idf * tf * (BM25_K1 + 1) / (tf + norm)


def test_bm25_ranking_matches_hand_computed_scores() -> None:
    chunks = [
        chunk("doc-a#chunk-000", "apple banana apple"),
        chunk("doc-b#chunk-000", "banana cherry"),
        chunk("doc-c#chunk-000", "cherry durian elderberry fig"),
    ]
    index = BM25Index.from_chunks(chunks)

    hits = index.search("apple cherry", top_k=3)

    # N=3, avgdl=(3+2+4)/3=3; apple df=1, cherry df=2; k1=1.2, b=0.75.
    expected = {
        "doc-a#chunk-000": hand_bm25(2, 3, 3.0, 1, 3),
        "doc-b#chunk-000": hand_bm25(1, 2, 3.0, 2, 3),
        "doc-c#chunk-000": hand_bm25(1, 4, 3.0, 2, 3),
    }
    assert [hit.chunk.id for hit in hits] == [
        "doc-a#chunk-000",
        "doc-b#chunk-000",
        "doc-c#chunk-000",
    ]
    for hit in hits:
        assert hit.score == pytest.approx(expected[hit.chunk.id], rel=1e-12)
    # Literal hand-calculated values guard against a shared formula mistake.
    assert [round(hit.score, 6) for hit in hits] == [1.34864, 0.544215, 0.413603]


def test_bm25_is_deterministic_and_breaks_ties_by_chunk_id() -> None:
    chunks = [
        chunk("doc-b#chunk-000", "rerank passages"),
        chunk("doc-a#chunk-000", "rerank passages"),
        chunk("doc-c#chunk-000", "unrelated words"),
    ]

    first = BM25Index.from_chunks(chunks).search("rerank", top_k=3)
    second = BM25Index.from_chunks(list(reversed(chunks))).search("rerank", top_k=3)

    assert [hit.chunk.id for hit in first] == [
        "doc-a#chunk-000",
        "doc-b#chunk-000",
        "doc-c#chunk-000",
    ]
    assert [(hit.chunk.id, hit.score) for hit in first] == [
        (hit.chunk.id, hit.score) for hit in second
    ]
    assert first[2].score == 0.0


def test_tokenizer_splits_latin_words_and_cjk_bigrams() -> None:
    assert tokenize("BM25检索 增强生成, RAG!") == [
        "bm25",
        "检索",
        "增强",
        "强生",
        "生成",
        "rag",
    ]
    assert tokenize("的") == ["的"]
    assert tokenize("Ｒｅｒａｎｋ") == ["rerank"]


def test_bm25_ranks_chinese_passages_by_shared_bigrams() -> None:
    chunks = [
        chunk("doc-a#chunk-000", "向量检索依赖嵌入模型"),
        chunk("doc-b#chunk-000", "关键词检索使用倒排索引"),
        chunk("doc-c#chunk-000", "生成模型负责回答问题"),
    ]

    hits = BM25Index.from_chunks(chunks).search("倒排索引", top_k=1)

    assert hits[0].chunk.id == "doc-b#chunk-000"


def test_bm25_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        BM25Index.from_chunks([chunk("doc-a#chunk-000", "text")]).search("text", top_k=0)


class RecordingEmbeddingProvider:
    """Local embedding that records every batch it is asked to embed."""

    def __init__(self) -> None:
        self.inner = FakeEmbeddingProvider(dimensions=16)
        self.cache_identity = self.inner.cache_identity
        self.batches: list[list[str]] = []

    def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        self.batches.append(list(texts))
        return self.inner.embed(texts, model=model)


def bm25_config(tmp_path: Path, *, embedding_model: str = "fake-hash-16") -> ExperimentConfig:
    corpus = tmp_path / "knowledge_base"
    corpus.mkdir(exist_ok=True)
    (corpus / "doc-01-rag.md").write_text(
        "# RAG\n\nID: doc-01\n\nRAG retrieves evidence.", encoding="utf-8"
    )
    (corpus / "doc-02-chunk.md").write_text(
        "# Chunking\n\nID: doc-02\n\nChunking splits documents.", encoding="utf-8"
    )
    return ExperimentConfig(
        name="bm25",
        mode="mock",
        dataset_path=tmp_path / "dataset.json",
        knowledge_base_path=corpus,
        database_path=tmp_path / "runs.sqlite3",
        artifact_dir=tmp_path / "artifacts",
        max_workers=1,
        provider=ProviderConfig(
            name="fake",
            base_url="https://offline.invalid/v1",
            api_key_env="UNUSED",
            chat_model="fake-chat",
            embedding_model=embedding_model,
        ),
        retrieval=[
            RetrievalConfig(
                chunk_size=200,
                chunk_overlap=20,
                top_k=1,
                prompt_variant="direct",
                retriever="bm25",
            )
        ],
        budget=BudgetConfig(hard_limit=20),
    )


def bm25_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        version="1.0.0",
        name="bm25",
        cases=[
            EvaluationCase.answerable(
                id="rag-001",
                question="What splits documents?",
                reference_answer="Chunking splits documents.",
                expected_document_ids=["doc-02"],
                reference_evidence=["Chunking splits documents."],
                category="chunking",
                difficulty="easy",
            )
        ],
    )


def test_bm25_is_selected_per_arm_and_never_embeds_chunks_or_queries(
    tmp_path: Path,
) -> None:
    config = bm25_config(tmp_path)
    cases = bm25_dataset()
    embedding = RecordingEmbeddingProvider()
    chat = FakeChatProvider(
        {
            case.question: StructuredAnswer(
                answer=case.reference_answer, citations=[], abstained=False
            )
            for case in cases.cases
        }
    )

    record = run_experiment(config, ProviderBundle(embedding=embedding, chat=chat), cases)

    assert record.status is ExperimentStatus.COMPLETED
    result = record.case_results[0]
    assert result.config_id == "chunk200-overlap20-top1-direct-bm25"
    assert result.retrieval_hits[0].chunk.document_id == "doc-02"
    assert result.metrics["retrieval_recall_at_k"] == 1.0
    # Only the answer-similarity metric embeds text; retrieval never does.
    assert embedding.batches == [["Chunking splits documents.", "Chunking splits documents."]]


def test_bm25_arm_is_absent_from_the_embedding_budget_plan(tmp_path: Path) -> None:
    config = bm25_config(tmp_path, embedding_model="remote-embedding")
    dense_arm = config.retrieval[0].model_copy(update={"retriever": "embedding"})
    mixed = config.model_copy(update={"retrieval": [config.retrieval[0], dense_arm]})
    documents = load_documents(config.knowledge_base_path)

    bm25_only = {call.phase: call for call in planned_calls(config, bm25_dataset(), documents)}
    mixed_plan = planned_calls(mixed, bm25_dataset(), documents)

    assert "embedding_index" not in bm25_only
    assert "embedding_query" not in bm25_only
    # The answer-similarity metric still embeds with the remote model.
    assert bm25_only["embedding_answer"].count == 1
    assert [call.phase for call in mixed_plan].count("embedding_index") == 1
    query_calls = [call for call in mixed_plan if call.phase == "embedding_query"]
    assert [call.count for call in query_calls] == [1]


def test_retrieval_config_id_is_unchanged_for_embedding_arms() -> None:
    arm = RetrievalConfig(chunk_size=300, chunk_overlap=50, top_k=4, prompt_variant="direct")
    bm25_arm = arm.model_copy(update={"retriever": "bm25"})

    assert arm.retriever == "embedding"
    assert arm.config_id == "chunk300-overlap50-top4-direct"
    assert bm25_arm.config_id == "chunk300-overlap50-top4-direct-bm25"
