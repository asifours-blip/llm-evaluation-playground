"""Document- and evidence-level retrieval metrics."""

from collections.abc import Collection, Sequence

from rag_quality_lab.domain.models import RetrievalHit


def recall_at_k(
    hits: Sequence[RetrievalHit], expected_document_ids: Collection[str], *, k: int
) -> float:
    """Return the fraction of expected documents found in the first K hits."""

    if k <= 0:
        raise ValueError("k must be positive")
    expected = set(expected_document_ids)
    if not expected:
        return 0.0
    retrieved = {hit.chunk.document_id for hit in hits[:k]}
    return len(retrieved & expected) / len(expected)


def reciprocal_rank(
    hits: Sequence[RetrievalHit], expected_document_ids: Collection[str]
) -> float:
    """Return the reciprocal rank of the first expected document."""

    expected = set(expected_document_ids)
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.document_id in expected:
            return 1.0 / rank
    return 0.0


def context_hit_rate(
    hits: Sequence[RetrievalHit], reference_evidence: Sequence[str]
) -> float:
    """Return the fraction of reference snippets present in retrieved text."""

    if not reference_evidence:
        return 0.0
    context = "\n".join(hit.chunk.text for hit in hits)
    found = sum(evidence in context for evidence in reference_evidence)
    return found / len(reference_evidence)
