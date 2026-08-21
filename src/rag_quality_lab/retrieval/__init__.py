"""Small-corpus retrieval primitives."""

from rag_quality_lab.retrieval.index import (
    InMemoryIndex,
    chunk_document,
    cosine_similarity,
    load_documents,
)

__all__ = ["InMemoryIndex", "chunk_document", "cosine_similarity", "load_documents"]
