import pytest

from rag_quality_lab.metrics.answer import (
    bilingual_f1,
    normalized_exact_match,
    semantic_similarity,
)


def test_normalized_exact_match_ignores_case_spacing_and_punctuation() -> None:
    assert normalized_exact_match(" RAG, works! ", "rag works") == 1.0


def test_bilingual_f1_uses_english_words_and_chinese_characters() -> None:
    assert bilingual_f1("RAG 检索 evidence", "rag 检索 facts") == pytest.approx(0.75)


def test_bilingual_f1_defines_empty_answers() -> None:
    assert bilingual_f1("", "") == 1.0
    assert bilingual_f1("answer", "") == 0.0


def test_semantic_similarity_uses_cosine() -> None:
    assert semantic_similarity([1.0, 0.0], [0.5, 0.0]) == pytest.approx(1.0)
