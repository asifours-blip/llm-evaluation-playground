"""Deterministic Okapi BM25 lexical retrieval baseline.

Tokenization
------------
Text is NFKC-normalized and case-folded (so full-width Latin letters and
digits match their ASCII forms). Each maximal run of CJK ideographs (CJK
Unified Ideographs, Extension A, and Compatibility Ideographs) becomes its
overlapping character bigrams, or the single character when the run has one
character; this is the dictionary-free scheme of Lucene's CJK analyzer, so
Chinese text needs no word segmenter or model. Every other maximal run of
Unicode letters and digits is one token. Punctuation, whitespace, and
underscores separate tokens. There is no stemming and no stop-word list.

Scoring
-------
``score(q, d) = sum over distinct query terms t of
idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))``
with ``idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))`` (non-negative),
``k1 = 1.2`` and ``b = 0.75``. Query terms are summed in sorted order and ties
are broken by chunk ID, so identical inputs always produce identical hits.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from rag_quality_lab.domain.models import Chunk, RetrievalHit

BM25_K1 = 1.2
BM25_B = 0.75
_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_TOKEN_PATTERN = re.compile(rf"(?P<cjk>[{_CJK}]+)|(?P<word>[^\W_{_CJK}]+)")


def tokenize(text: str) -> list[str]:
    """Split text into Latin/other word tokens and CJK character bigrams."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        run = match.group("cjk")
        if run is None:
            tokens.append(match.group("word"))
        elif len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class BM25Index:
    """In-memory BM25 index over a fixed chunk list."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._term_counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        total = sum(self._lengths)
        self._average_length = total / len(self._lengths) if total else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        self._document_frequency = document_frequency

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> BM25Index:
        return cls(chunks)

    def idf(self, term: str) -> float:
        count = len(self.chunks)
        frequency = self._document_frequency.get(term, 0)
        return math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        """Return the ``top_k`` highest-scoring chunks, ties ordered by chunk ID."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = sorted(set(tokenize(query)))
        weights = [(term, self.idf(term)) for term in terms]
        hits = [
            RetrievalHit(chunk=chunk, score=self._score(weights, counts, length))
            for chunk, counts, length in zip(
                self.chunks, self._term_counts, self._lengths, strict=True
            )
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]

    def _score(
        self,
        weights: Sequence[tuple[str, float]],
        counts: Counter[str],
        length: int,
    ) -> float:
        if not self._average_length:
            return 0.0
        norm = BM25_K1 * (1 - BM25_B + BM25_B * length / self._average_length)
        score = 0.0
        for term, idf in weights:
            frequency = counts.get(term, 0)
            if frequency:
                score += idf * frequency * (BM25_K1 + 1) / (frequency + norm)
        return score
