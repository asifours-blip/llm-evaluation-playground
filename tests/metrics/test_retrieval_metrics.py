import pytest

from rag_quality_lab.domain.models import Chunk, RetrievalHit
from rag_quality_lab.metrics.retrieval import context_hit_rate, recall_at_k, reciprocal_rank


def hit(document_id: str, rank: int, text: str = "") -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(
            id=f"{document_id}#chunk-{rank:03d}",
            document_id=document_id,
            text=text,
        ),
        score=1 / rank,
    )


def test_recall_and_mrr_use_stable_document_ids() -> None:
    hits = [hit("doc-x", 1), hit("doc-b", 2), hit("doc-a", 3)]

    assert recall_at_k(hits, {"doc-a", "doc-b"}, k=2) == 0.5
    assert reciprocal_rank(hits, {"doc-a", "doc-b"}) == 0.5


def test_context_hit_rate_counts_reference_evidence_in_chunks() -> None:
    hits = [hit("doc-a", 1, "Alpha evidence appears here."), hit("doc-b", 2, "Other")]

    assert context_hit_rate(hits, ["Alpha evidence", "Missing evidence"]) == 0.5


def test_retrieval_metrics_define_empty_ground_truth_as_zero() -> None:
    assert recall_at_k([], set(), k=1) == 0.0
    assert reciprocal_rank([], set()) == 0.0
    assert context_hit_rate([], []) == 0.0


def test_recall_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k"):
        recall_at_k([], {"doc-a"}, k=0)
