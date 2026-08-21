import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_quality_lab.providers.openai_compatible import (
    AuthenticationError,
    OpenAICompatibleProvider,
    ProviderError,
)


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def answer_response(content: str) -> FakeResponse:
    return FakeResponse(
        status_code=200,
        payload={
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "prompt_cache_hit_tokens": 6,
                "prompt_cache_miss_tokens": 4,
            },
        },
    )


@pytest.fixture
def make_provider(monkeypatch: pytest.MonkeyPatch) -> Callable[..., OpenAICompatibleProvider]:
    monkeypatch.setenv("TEST_API_KEY", "secret-key")

    def factory(**kwargs: Any) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            base_url="https://example.com/v1",
            api_key_env="TEST_API_KEY",
            max_retries=1,
            jitter=lambda: 0.0,
            **kwargs,
        )

    return factory


def test_authentication_failure_is_not_retried(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession([FakeResponse(401, {"error": "bad key"}, text="bad key")])
    provider = make_provider(session=session)

    with pytest.raises(AuthenticationError):
        provider.answer("q", [], model="m")

    assert len(session.calls) == 1


def test_rate_limit_honors_retry_after(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    sleeps: list[float] = []
    session = FakeSession(
        [
            FakeResponse(
                429,
                {"error": "slow"},
                headers={"Retry-After": "2"},
                text="slow",
            ),
            answer_response('{"answer":"A","citations":[],"abstained":false}'),
        ]
    )
    provider = make_provider(session=session, sleeper=sleeps.append)

    provider.answer("q", [], model="m")

    assert sleeps == [2.0]
    assert len(session.calls) == 2


def test_error_never_contains_api_key(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    provider = make_provider(session=FakeSession([]))

    with pytest.raises(ProviderError) as error:
        provider._raise_sanitized("Authorization: Bearer secret-key")

    assert "secret-key" not in str(error.value)


def test_answer_parses_usage_and_provider_specific_extra_body(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession(
        [answer_response('{"answer":"A","citations":["doc-01"],"abstained":false}')]
    )
    provider = make_provider(
        session=session,
        extra_body={
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "top_p": 1,
        },
    )

    response = provider.answer("q", ["context"], model="m")

    assert response.parsed.citations == ["doc-01"]
    assert response.usage.input_cache_hit_tokens == 6
    assert response.usage.input_cache_miss_tokens == 4
    request_json = session.calls[0]["json"]
    assert request_json["response_format"] == {"type": "json_object"}
    assert request_json["thinking"] == {"type": "disabled"}
    assert request_json["temperature"] == 0
    assert request_json["top_p"] == 1
    assert request_json["max_tokens"] == 512


def test_answer_bounds_serialized_input_before_sending(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession(
        [answer_response('{"answer":"A","citations":[],"abstained":false}')]
    )
    provider = make_provider(session=session)

    provider.answer("q", ["x" * 20_000], model="m")

    messages = session.calls[0]["json"]["messages"]
    serialized_size = len(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
    )
    assert serialized_size <= 2372


def test_malformed_answer_gets_one_repair_attempt(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession(
        [
            answer_response("not json"),
            answer_response('{"answer":"fixed","citations":[],"abstained":false}'),
        ]
    )
    provider = make_provider(session=session)

    response = provider.answer("q", [], model="m")

    assert response.parsed.answer == "fixed"
    assert len(session.calls) == 2


def test_answer_contract_and_repair_forbid_null_answer(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession(
        [
            answer_response('{"answer":null,"citations":[],"abstained":false}'),
            answer_response('{"answer":"fixed","citations":[],"abstained":false}'),
        ]
    )
    provider = make_provider(session=session)

    response = provider.answer("q", [], model="m")

    assert response.parsed.answer == "fixed"
    assert len(session.calls) == 2
    assert "non-empty string" in session.calls[0]["json"]["messages"][0]["content"]
    assert "must not be null" in session.calls[1]["json"]["messages"][1]["content"]


def test_judge_returns_structured_verdict_and_usage(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession(
        [answer_response('{"score":5,"passed":true,"reason":"fully supported"}')]
    )
    provider = make_provider(session=session)

    response = provider.judge(
        "q",
        "reference",
        "candidate",
        ["evidence"],
        model="judge-model",
    )

    assert response.parsed.score == 5
    assert response.usage.total_tokens == 14
    request = session.calls[0]["json"]
    assert request["model"] == "judge-model"
    assert request["max_tokens"] == 256
    assert "Score the candidate" in request["messages"][1]["content"]


def test_pairwise_judge_returns_structured_preference(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession(
        [answer_response('{"preferred":"A","reason":"more faithful"}')]
    )
    provider = make_provider(session=session)

    response = provider.pairwise(
        "q",
        "reference",
        ["evidence"],
        "first",
        "second",
        model="judge-model",
    )

    assert response.parsed.preferred == "A"
    assert response.usage.total_tokens == 14
    assert "A:\nfirst" in session.calls[0]["json"]["messages"][1]["content"]


def test_embeddings_preserve_response_order(
    make_provider: Callable[..., OpenAICompatibleProvider],
) -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )
        ]
    )
    provider = make_provider(session=session)

    vectors = provider.embed(["first", "second"], model="embedding-model")

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
