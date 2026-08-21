"""Deterministic in-memory retrieval for small controlled corpora."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

from rag_quality_lab.domain.models import Chunk, Document, RetrievalHit
from rag_quality_lab.providers.base import EmbeddingProvider


def chunk_document(
    document_id: str,
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split text with a deterministic character window."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    for index, start in enumerate(range(0, len(text), step)):
        end = min(start + chunk_size, len(text))
        chunks.append(
            Chunk(
                id=f"{document_id}#chunk-{index:03d}",
                document_id=document_id,
                text=text[start:end],
                start_char=start,
                end_char=end,
            )
        )
        if end == len(text):
            break
    return chunks


def load_documents(directory: str | Path) -> list[Document]:
    """Load Markdown documents containing a heading and `ID: doc-NN` line."""

    documents: list[Document] = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        title = next((line[2:].strip() for line in lines if line.startswith("# ")), "")
        document_id = next(
            (line.removeprefix("ID:").strip() for line in lines if line.startswith("ID:")),
            "",
        )
        if not title or not document_id:
            raise ValueError(f"document requires title and stable ID: {path}")
        documents.append(
            Document(
                id=document_id,
                title=title,
                text=text,
                source_path=str(path),
            )
        )
    document_ids = [document.id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("knowledge base document IDs must be unique")
    return documents


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity without a numerical dependency."""

    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)


class InMemoryIndex:
    """Brute-force cosine index with stable ordering."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        provider: EmbeddingProvider,
        model: str | None,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal lengths")
        self.chunks = list(chunks)
        self.embeddings = [list(vector) for vector in embeddings]
        self.provider = provider
        self.model = model

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        provider: EmbeddingProvider,
        *,
        model: str | None = None,
        cache_path: str | Path | None = None,
    ) -> InMemoryIndex:
        chunk_list = list(chunks)
        cache_file = Path(cache_path) if cache_path is not None else None
        cache = _load_cache(cache_file) if cache_file is not None else {}
        chunking_hash = _hash_text(
            "\n".join(f"{chunk.id}\0{chunk.text}" for chunk in chunk_list)
        )
        provider_id = str(
            getattr(
                provider,
                "cache_identity",
                f"{type(provider).__module__}.{type(provider).__qualname__}",
            )
        )
        keys = [
            _cache_key(provider_id, model, chunk, chunking_hash) for chunk in chunk_list
        ]

        missing_indexes = [index for index, key in enumerate(keys) if key not in cache]
        if missing_indexes:
            missing_texts = [chunk_list[index].text for index in missing_indexes]
            missing_vectors = provider.embed(missing_texts, model=model)
            if len(missing_vectors) != len(missing_indexes):
                raise ValueError("embedding provider returned an unexpected vector count")
            for index, vector in zip(missing_indexes, missing_vectors, strict=True):
                cache[keys[index]] = vector
            if cache_file is not None:
                _write_cache(cache_file, cache)

        embeddings = [cache[key] for key in keys]
        return cls(chunk_list, embeddings, provider, model)

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vectors = self.provider.embed([query], model=self.model)
        if len(query_vectors) != 1:
            raise ValueError("embedding provider must return one query vector")
        hits = [
            RetrievalHit(chunk=chunk, score=cosine_similarity(query_vectors[0], vector))
            for chunk, vector in zip(self.chunks, self.embeddings, strict=True)
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]


def _cache_key(
    provider_id: str,
    model: str | None,
    chunk: Chunk,
    chunking_hash: str,
) -> str:
    return _hash_text(
        "\0".join(
            [
                _hash_text(provider_id),
                _hash_text(model or ""),
                _hash_text(chunk.text),
                chunking_hash,
                chunk.id,
            ]
        )
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("embedding cache must contain a JSON object")
    return {
        str(key): [float(value) for value in vector]
        for key, vector in payload.items()
        if isinstance(vector, list)
    }


def _write_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(cache, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)
