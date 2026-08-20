"""Provider interfaces consumed by the experiment runner."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag_quality_lab.domain.models import ProviderResponse, StructuredAnswer


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Convert text batches into vectors."""

    def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        """Return one vector per input text."""


@runtime_checkable
class ChatProvider(Protocol):
    """Generate structured answers from questions and retrieved context."""

    def answer(
        self,
        question: str,
        contexts: Sequence[str],
        *,
        model: str,
    ) -> ProviderResponse[StructuredAnswer]:
        """Return a structured answer and usage metadata."""
