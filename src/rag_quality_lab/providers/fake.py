"""Deterministic providers for offline evaluation and CI."""

import hashlib
import math
from collections.abc import Mapping, Sequence

from rag_quality_lab.domain.models import ProviderResponse, StructuredAnswer, TokenUsage


class FakeEmbeddingProvider:
    """Create stable normalized vectors by hashing whitespace tokens."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        del model
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class FakeChatProvider:
    """Return explicit scripted answers without network access."""

    def __init__(self, answers: Mapping[str, StructuredAnswer]) -> None:
        self.answers = dict(answers)

    def answer(
        self,
        question: str,
        contexts: Sequence[str],
        *,
        model: str,
    ) -> ProviderResponse[StructuredAnswer]:
        if question not in self.answers:
            raise KeyError(f"Unscripted question: {question}")
        parsed = self.answers[question].model_copy(deep=True)
        input_text = " ".join([question, *contexts])
        input_tokens = max(1, len(input_text.encode("utf-8")) // 4)
        output_tokens = max(1, len(parsed.model_dump_json().encode("utf-8")) // 4)
        return ProviderResponse[StructuredAnswer](
            parsed=parsed,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            model=model,
        )
