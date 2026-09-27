"""Shared harness for interruption, resume, and cancellation tests.

The harness drives the real ``OpenAICompatibleProvider`` through an in-process
HTTP double. Every physical request is appended to a JSONL log and flushed to
disk before the double answers, so a parent test can count requests made by a
child process that it later kills. No network access or real API key is used.

Run as a script, it executes one live experiment and can freeze itself at a
chosen point so the parent can kill it:

* ``--block-after-results K`` freezes right after the K-th case outcome is
  committed, when no request is in flight.
* ``--block-generation-case CASE_ID`` freezes inside the generation request of
  one case, after the request was sent but before any response is settled.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import yaml

API_KEY_ENV = "RAG_QUALITY_TEST_ONLY_KEY"
EMBEDDING_MODEL = "remote-embedding"
CHAT_MODEL = "remote-chat"
JUDGE_MODEL = "remote-judge"
CORPUS_TEXT = (
    "# RAG\n\nID: doc-01\n\nRAG retrieves evidence. Chunking splits documents. "
    "Reranking orders passages."
)
JUDGE_MAX_TOKENS = 256
HARNESS_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = HARNESS_PATH.parents[2]

CASES: list[dict[str, Any]] = [
    {
        "id": f"rag-00{index}",
        "question": question,
        "reference_answer": answer,
        "answerability": "answerable",
        "expected_document_ids": ["doc-01"],
        "reference_evidence": [evidence],
        "category": "retrieval",
        "difficulty": "easy",
    }
    for index, (question, answer, evidence) in enumerate(
        [
            ("What does RAG retrieve?", "RAG retrieves evidence.", "RAG retrieves evidence."),
            ("What does chunking do?", "Chunking splits documents.", "Chunking splits"),
            ("What does reranking do?", "Reranking orders passages.", "Reranking orders"),
            ("Which step retrieves evidence?", "Retrieval.", "RAG retrieves evidence."),
            ("What splits documents?", "Chunking.", "Chunking splits documents."),
        ],
        start=1,
    )
]


def default_prices(rate: float = 1) -> dict[str, dict[str, float]]:
    return {
        CHAT_MODEL: {"input_cache_miss": rate, "output": rate},
        JUDGE_MODEL: {"input_cache_miss": rate, "output": rate},
        EMBEDDING_MODEL: {"input_cache_miss": rate, "output": 0},
    }


def generation_only_prices(output_rate: float) -> dict[str, dict[str, float]]:
    """Price only generation output so each case costs exactly its reservation."""

    return {
        CHAT_MODEL: {"input_cache_miss": 0, "output": output_rate},
        JUDGE_MODEL: {"input_cache_miss": 0, "output": 0},
        EMBEDDING_MODEL: {"input_cache_miss": 0, "output": 0},
    }


def write_inputs(
    workdir: Path,
    *,
    prices: dict[str, dict[str, float]] | None = None,
    hard_limit: str = "20",
    preflight_fraction: str = "0.90",
    safety_multiplier: str = "1.25",
    max_retries: int = 1,
    verified_at: date | None = None,
) -> Path:
    """Write corpus, dataset, pricing, and a live config; return the config path."""

    workdir.mkdir(parents=True, exist_ok=True)
    corpus = workdir / "knowledge_base"
    corpus.mkdir(exist_ok=True)
    (corpus / "doc-01-rag.md").write_text(CORPUS_TEXT, encoding="utf-8")
    dataset = workdir / "dataset.json"
    dataset.write_text(
        json.dumps({"version": "1.0.0", "name": "resume-harness", "cases": CASES}),
        encoding="utf-8",
    )
    pricing = workdir / "pricing.yaml"
    pricing.write_text(
        yaml.safe_dump(
            {
                "provider": "local-test",
                "currency": "CNY",
                "verified_at": (verified_at or date.today()).isoformat(),
                "source_url": "https://example.com/pricing",
                "models": prices or default_prices(),
            }
        ),
        encoding="utf-8",
    )
    config = workdir / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "resume-harness",
                "mode": "live",
                "dataset_path": str(dataset),
                "knowledge_base_path": str(corpus),
                "database_path": str(workdir / "experiments.sqlite3"),
                "artifact_dir": str(workdir / "artifacts"),
                "max_workers": 1,
                "provider": {
                    "name": "local-test",
                    "base_url": "https://provider.test/v1",
                    "api_key_env": API_KEY_ENV,
                    "chat_model": CHAT_MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "max_retries": max_retries,
                },
                "retrieval": [
                    {
                        "chunk_size": 200,
                        "chunk_overlap": 20,
                        "top_k": 1,
                        "prompt_variant": "direct",
                    }
                ],
                "budget": {
                    "currency": "CNY",
                    "hard_limit": hard_limit,
                    "preflight_fraction": preflight_fraction,
                    "safety_multiplier": safety_multiplier,
                },
                "pricing_path": str(pricing),
            }
        ),
        encoding="utf-8",
    )
    return config


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers: dict[str, str] = {}
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self.payload


RequestHook = Callable[[dict[str, Any]], None]


class LoggedSession:
    """OpenAI-compatible double that durably logs each request before replying."""

    def __init__(
        self,
        log_path: Path,
        *,
        generation_completion_tokens: int = 20,
        on_request: RequestHook | None = None,
    ) -> None:
        from rag_quality_lab.providers.fake import FakeEmbeddingProvider

        self.log_path = log_path
        self.generation_completion_tokens = generation_completion_tokens
        self.on_request = on_request
        self.vectors = FakeEmbeddingProvider(dimensions=32)

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _Response:
        del headers, timeout
        path = url.split("/v1", maxsplit=1)[1]
        entry = classify(path, json)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(_json_line(entry))
            handle.flush()
            os.fsync(handle.fileno())
        if self.on_request is not None:
            self.on_request(entry)
        if path == "/embeddings":
            texts = list(json["input"])
            tokens = sum(max(1, len(text.encode("utf-8")) // 4) for text in texts)
            return _Response(
                200,
                {
                    "data": [
                        {"index": index, "embedding": vector}
                        for index, vector in enumerate(self.vectors.embed(texts))
                    ],
                    "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                },
            )
        case = _case(entry["case_id"])
        if entry["kind"] == "judge":
            content: dict[str, Any] = {"score": 4, "passed": True, "reason": "grounded"}
            completion_tokens = 10
        else:
            content = {
                "answer": case["reference_answer"],
                "citations": ["doc-01"],
                "abstained": False,
            }
            completion_tokens = self.generation_completion_tokens
        return _Response(
            200,
            {
                "choices": [{"message": {"content": _dumps(content)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": completion_tokens},
            },
        )


def classify(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Name the pipeline step and case that one request belongs to."""

    if path == "/embeddings":
        texts = list(body["input"])
        for case in CASES:
            if texts == [case["question"]]:
                return {"path": path, "kind": "embedding_query", "case_id": case["id"]}
            if len(texts) == 2 and texts[1] == case["reference_answer"]:
                return {"path": path, "kind": "embedding_answer", "case_id": case["id"]}
        return {"path": path, "kind": "embedding_index", "case_id": None}
    messages = " ".join(str(message.get("content", "")) for message in body["messages"])
    kind = "judge" if body.get("max_tokens") == JUDGE_MAX_TOKENS else "generation"
    for case in CASES:
        if case["question"] in messages:
            return {"path": path, "kind": kind, "case_id": case["id"]}
    return {"path": path, "kind": kind, "case_id": None}


def read_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def remote_bundle(session: LoggedSession, *, max_retries: int = 1) -> Any:
    from rag_quality_lab.experiments.runner import ProviderBundle
    from rag_quality_lab.providers.openai_compatible import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="https://provider.test/v1",
        api_key_env=API_KEY_ENV,
        max_retries=max_retries,
        session=session,
        sleeper=lambda _: None,
        jitter=lambda: 0.0,
    )
    return ProviderBundle(embedding=provider, chat=provider, judge=provider)


def start_child(
    config: Path,
    log_path: Path,
    marker: Path,
    *extra: str,
    generation_completion_tokens: int = 20,
) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment[API_KEY_ENV] = "test-only-not-a-real-key"
    return subprocess.Popen(
        [
            sys.executable,
            str(HARNESS_PATH),
            "--config",
            str(config),
            "--log",
            str(log_path),
            "--marker",
            str(marker),
            "--generation-completion-tokens",
            str(generation_completion_tokens),
            *extra,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def kill_when_frozen(
    process: subprocess.Popen[str], marker: Path, *, timeout: float = 120
) -> None:
    """Wait until the child reports it is frozen, then kill it without cleanup."""

    deadline = time.monotonic() + timeout
    while not marker.exists():
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise AssertionError(f"child exited before freezing: {stderr}")
        if time.monotonic() > deadline:
            process.kill()
            raise AssertionError("child did not freeze in time")
        time.sleep(0.05)
    process.kill()
    process.communicate()


def _freeze(marker: Path) -> None:
    marker.write_text("frozen", encoding="utf-8")
    time.sleep(3600)


def _case(case_id: str | None) -> dict[str, Any]:
    return next(case for case in CASES if case["id"] == case_id)


def _dumps(payload: object) -> str:
    return json.dumps(payload)


def _json_line(payload: object) -> str:
    return json.dumps(payload, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--generation-completion-tokens", type=int, default=20)
    parser.add_argument("--block-after-results", type=int)
    parser.add_argument("--block-generation-case")
    args = parser.parse_args()

    from rag_quality_lab.config import load_dataset, load_experiment_config
    from rag_quality_lab.experiments.runner import run_experiment
    from rag_quality_lab.experiments.store import ExperimentStore

    if args.block_after_results is not None:
        original = ExperimentStore.commit_case_outcome
        committed = 0

        def freezing_commit(self: Any, *positional: Any, **named: Any) -> None:
            nonlocal committed
            original(self, *positional, **named)
            committed += 1
            if committed == args.block_after_results:
                _freeze(args.marker)

        ExperimentStore.commit_case_outcome = freezing_commit  # type: ignore[method-assign]

    def hook(entry: dict[str, Any]) -> None:
        if (
            args.block_generation_case is not None
            and entry["kind"] == "generation"
            and entry["case_id"] == args.block_generation_case
        ):
            _freeze(args.marker)

    config = load_experiment_config(args.config)
    session = LoggedSession(
        args.log,
        generation_completion_tokens=args.generation_completion_tokens,
        on_request=hook,
    )
    record = run_experiment(
        config,
        remote_bundle(session, max_retries=config.provider.max_retries),
        load_dataset(config.dataset_path),
    )
    print(json.dumps({"experiment_id": record.id, "status": record.status.value}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
