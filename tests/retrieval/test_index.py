from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_quality_lab.domain.models import Chunk
from rag_quality_lab.retrieval.index import InMemoryIndex, chunk_document, load_documents


class ConstantEmbeddingProvider:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls = 0

    def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        del model
        self.calls += 1
        return [self.vector.copy() for _ in texts]


def test_chunk_ids_are_stable() -> None:
    chunks = chunk_document("doc-01", "abcdef", chunk_size=4, chunk_overlap=1)

    assert [chunk.id for chunk in chunks] == [
        "doc-01#chunk-000",
        "doc-01#chunk-001",
    ]
    assert [chunk.text for chunk in chunks] == ["abcd", "def"]


def test_chunking_rejects_overlap_equal_to_size() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_document("doc-01", "abcdef", chunk_size=4, chunk_overlap=4)


def test_search_breaks_score_ties_by_chunk_id() -> None:
    provider = ConstantEmbeddingProvider([1.0, 0.0])
    index = InMemoryIndex.from_chunks(
        [
            Chunk(id="doc-02#chunk-000", document_id="doc-02", text="B"),
            Chunk(id="doc-01#chunk-000", document_id="doc-01", text="A"),
        ],
        provider,
    )

    hits = index.search("query", top_k=2)

    assert [hit.chunk.id for hit in hits] == [
        "doc-01#chunk-000",
        "doc-02#chunk-000",
    ]


def test_search_rejects_non_positive_top_k() -> None:
    index = InMemoryIndex.from_chunks(
        [Chunk(id="doc-01#chunk-000", document_id="doc-01", text="A")],
        ConstantEmbeddingProvider([1.0]),
    )

    with pytest.raises(ValueError, match="top_k"):
        index.search("query", top_k=0)


def test_embedding_cache_keys_include_model(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.json"
    chunks = [Chunk(id="doc-01#chunk-000", document_id="doc-01", text="A")]
    provider = ConstantEmbeddingProvider([1.0])

    InMemoryIndex.from_chunks(chunks, provider, model="model-a", cache_path=cache_path)
    InMemoryIndex.from_chunks(chunks, provider, model="model-a", cache_path=cache_path)
    assert provider.calls == 1

    InMemoryIndex.from_chunks(chunks, provider, model="model-b", cache_path=cache_path)
    assert provider.calls == 2


def test_load_documents_reads_stable_id_and_title(tmp_path: Path) -> None:
    path = tmp_path / "doc-01-example.md"
    path.write_text("# Example Title\n\nID: doc-01\n\nBody.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents[0].id == "doc-01"
    assert documents[0].title == "Example Title"
    assert documents[0].source_path == str(path)
