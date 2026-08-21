"""Deterministic answer-quality metrics."""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from rag_quality_lab.retrieval.index import cosine_similarity

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")


def normalize_answer(value: str) -> str:
    """Normalize Unicode, case, punctuation, and whitespace for comparison."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        character if character.isalnum() else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def bilingual_f1(prediction: str, reference: str) -> float:
    """Calculate token F1 using English words and individual CJK characters."""

    prediction_tokens = TOKEN_PATTERN.findall(normalize_answer(prediction))
    reference_tokens = TOKEN_PATTERN.findall(normalize_answer(reference))
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(
    prediction_embedding: Sequence[float], reference_embedding: Sequence[float]
) -> float:
    return cosine_similarity(prediction_embedding, reference_embedding)
