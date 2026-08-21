"""Hardened OpenAI-compatible chat and embedding provider."""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn, Protocol

import requests
from pydantic import ValidationError

from rag_quality_lab.domain.models import (
    JudgeVerdict,
    ProviderResponse,
    StructuredAnswer,
    TokenUsage,
)
from rag_quality_lab.metrics.judge import build_scalar_judge_prompt, parse_judge_verdict

MAX_ERROR_LENGTH = 500


class HTTPResponse(Protocol):
    @property
    def status_code(self) -> int:
        """HTTP status code."""

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""

    @property
    def text(self) -> str:
        """Response body text."""

    def json(self) -> object:
        """Decode the response body."""


class HTTPSession(Protocol):
    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HTTPResponse:
        """Submit one JSON request."""


class RequestsSessionAdapter:
    """Expose requests.Session through the narrow provider HTTP interface."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> HTTPResponse:
        response = self.session.post(url, json=json, headers=headers, timeout=timeout)
        return RequestsResponseAdapter(response)


class RequestsResponseAdapter:
    """Normalize requests.Response attributes for the provider boundary."""

    def __init__(self, response: requests.Response) -> None:
        self.response = response

    @property
    def status_code(self) -> int:
        return self.response.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return dict(self.response.headers)

    @property
    def text(self) -> str:
        return self.response.text

    def json(self) -> object:
        return self.response.json()


class ProviderError(RuntimeError):
    """Sanitized provider failure safe for experiment persistence."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class AuthenticationError(ProviderError):
    """Non-retryable authentication or authorization failure."""


class OpenAICompatibleProvider:
    """Access chat completions and embeddings through an OpenAI-style API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 60,
        max_retries: int = 2,
        session: HTTPSession | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise AuthenticationError(f"missing API key environment variable: {api_key_env}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.session: HTTPSession = session or RequestsSessionAdapter()
        self.sleeper = sleeper
        self.jitter = jitter
        self.extra_body = dict(extra_body or {})

    def answer(
        self,
        question: str,
        contexts: Sequence[str],
        *,
        model: str,
        instructions: str | None = None,
    ) -> ProviderResponse[StructuredAnswer]:
        started_at = time.perf_counter()
        primary = self._chat_completion(
            model=model,
            messages=self._answer_messages(question, contexts, instructions),
        )
        usage = self._token_usage(primary)
        content = self._message_content(primary)
        final_payload = primary
        try:
            parsed = self._parse_answer(content)
        except (json.JSONDecodeError, TypeError, ValidationError):
            repaired = self._chat_completion(
                model=model,
                messages=self._repair_messages(content),
            )
            usage = _combine_usage(usage, self._token_usage(repaired))
            final_payload = repaired
            try:
                parsed = self._parse_answer(self._message_content(repaired))
            except (json.JSONDecodeError, TypeError, ValidationError) as error:
                self._raise_sanitized(f"structured answer validation failed: {error}")

        return ProviderResponse[StructuredAnswer](
            parsed=parsed,
            usage=usage,
            model=model,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            raw=final_payload,
        )

    def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        if not model:
            raise ValueError("embedding model is required")
        payload = self._post_json(
            "/embeddings",
            {"model": model, "input": list(texts)},
        )
        data = payload.get("data")
        if not isinstance(data, list):
            self._raise_sanitized("embedding response is missing data")
        ordered: list[tuple[int, list[float]]] = []
        for item in data:
            if not isinstance(item, dict):
                self._raise_sanitized("embedding response contains an invalid item")
            index = item.get("index")
            embedding = item.get("embedding")
            if not isinstance(index, int) or not isinstance(embedding, list):
                self._raise_sanitized("embedding response contains invalid fields")
            ordered.append((index, [float(value) for value in embedding]))
        ordered.sort(key=lambda item: item[0])
        if len(ordered) != len(texts):
            self._raise_sanitized("embedding response count does not match input")
        return [embedding for _, embedding in ordered]

    def judge(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        evidence: Sequence[str],
        *,
        model: str,
    ) -> ProviderResponse[JudgeVerdict]:
        """Score one answer using the fixed structured judge rubric."""

        started_at = time.perf_counter()
        prompt = build_scalar_judge_prompt(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            evidence=list(evidence),
        )
        primary = self._chat_completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Apply the supplied rubric. Return one JSON object only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        usage = self._token_usage(primary)
        final_payload = primary
        try:
            parsed = parse_judge_verdict(self._message_content(primary))
        except (json.JSONDecodeError, TypeError, ValidationError):
            repaired = self._chat_completion(
                model=model,
                messages=self._judge_repair_messages(self._message_content(primary)),
            )
            usage = _combine_usage(usage, self._token_usage(repaired))
            final_payload = repaired
            try:
                parsed = parse_judge_verdict(self._message_content(repaired))
            except (json.JSONDecodeError, TypeError, ValidationError) as error:
                self._raise_sanitized(f"structured judge validation failed: {error}")
        return ProviderResponse[JudgeVerdict](
            parsed=parsed,
            usage=usage,
            model=model,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            raw=final_payload,
        )

    def _chat_completion(
        self, *, model: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        body.update(self.extra_body)
        return self._post_json("/chat/completions", body)

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}{path}",
                    json=body,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as error:
                if attempt >= self.max_retries:
                    self._raise_sanitized(f"network request failed: {error}", retryable=True)
                self.sleeper(self._backoff(attempt, None))
                continue

            if response.status_code in {401, 403}:
                raise AuthenticationError(
                    self._sanitize(self._response_error(response)),
                    status_code=response.status_code,
                )
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable:
                if attempt >= self.max_retries:
                    self._raise_sanitized(
                        self._response_error(response),
                        retryable=True,
                        status_code=response.status_code,
                    )
                self.sleeper(self._backoff(attempt, response.headers.get("Retry-After")))
                continue
            if response.status_code >= 400:
                self._raise_sanitized(
                    self._response_error(response),
                    status_code=response.status_code,
                )
            payload = response.json()
            if not isinstance(payload, dict):
                self._raise_sanitized("provider response must be a JSON object")
            return payload
        raise AssertionError("retry loop must return or raise")

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return min(10.0, float(2**attempt)) + self.jitter()

    def _response_error(self, response: HTTPResponse) -> str:
        return response.text or f"provider returned HTTP {response.status_code}"

    def _sanitize(self, message: str) -> str:
        sanitized = message.replace(self.api_key, "[REDACTED]")
        sanitized = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", sanitized)
        return sanitized[:MAX_ERROR_LENGTH]

    def _raise_sanitized(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> NoReturn:
        raise ProviderError(
            self._sanitize(message),
            retryable=retryable,
            status_code=status_code,
        )

    @staticmethod
    def _answer_messages(
        question: str,
        contexts: Sequence[str],
        instructions: str | None,
    ) -> list[dict[str, str]]:
        context = "\n\n".join(contexts)
        system_message = instructions or (
            "Answer only from the supplied context. Return JSON with answer, "
            "citations, and abstained. Abstain when evidence is insufficient."
        )
        return [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ]

    @staticmethod
    def _repair_messages(content: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Repair the content into valid JSON only; do not add new facts.",
            },
            {
                "role": "user",
                "content": (
                    "Required keys: answer (string), citations (string array), "
                    f"abstained (boolean). Content:\n{content}"
                ),
            },
        ]

    @staticmethod
    def _judge_repair_messages(content: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Repair the content into valid JSON only; do not rescore it.",
            },
            {
                "role": "user",
                "content": (
                    "Required keys: score (integer 1-5), passed (boolean equal to "
                    f"score >= 4), reason (string). Content:\n{content}"
                ),
            },
        ]

    @staticmethod
    def _message_content(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError("chat response is missing message content") from error
        if not isinstance(content, str):
            raise ProviderError("chat response content must be text")
        return content

    @staticmethod
    def _parse_answer(content: str) -> StructuredAnswer:
        return StructuredAnswer.model_validate(json.loads(content))

    @staticmethod
    def _token_usage(payload: dict[str, Any]) -> TokenUsage:
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            raise ProviderError("chat response usage must be an object")
        return TokenUsage(
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            input_cache_hit_tokens=int(usage.get("prompt_cache_hit_tokens", 0)),
            input_cache_miss_tokens=int(usage.get("prompt_cache_miss_tokens", 0)),
        )


def _combine_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        input_cache_hit_tokens=(
            left.input_cache_hit_tokens + right.input_cache_hit_tokens
        ),
        input_cache_miss_tokens=(
            left.input_cache_miss_tokens + right.input_cache_miss_tokens
        ),
    )
