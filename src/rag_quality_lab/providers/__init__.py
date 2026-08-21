"""Provider protocols and implementations."""

from rag_quality_lab.providers.base import ChatProvider, EmbeddingProvider, JudgeProvider
from rag_quality_lab.providers.fake import (
    FakeChatProvider,
    FakeEmbeddingProvider,
    FakeJudgeProvider,
)
from rag_quality_lab.providers.openai_compatible import (
    AuthenticationError,
    OpenAICompatibleProvider,
    ProviderError,
)

__all__ = [
    "AuthenticationError",
    "ChatProvider",
    "EmbeddingProvider",
    "FakeChatProvider",
    "FakeEmbeddingProvider",
    "FakeJudgeProvider",
    "JudgeProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
]
