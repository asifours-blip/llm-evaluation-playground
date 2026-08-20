import json
import subprocess
import sys
from pathlib import Path

import yaml


def write_cli_fixture(tmp_path: Path, *, mode: str = "mock") -> Path:
    corpus = tmp_path / "knowledge_base"
    corpus.mkdir(parents=True)
    (corpus / "doc-01-rag.md").write_text(
        "# RAG\n\nID: doc-01\n\nRAG retrieves evidence.",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "name": "cli-fixture",
                "cases": [
                    {
                        "id": "rag-001",
                        "question": "What does RAG retrieve?",
                        "reference_answer": "RAG retrieves evidence.",
                        "answerability": "answerable",
                        "expected_document_ids": ["doc-01"],
                        "reference_evidence": ["RAG retrieves evidence."],
                        "category": "retrieval",
                        "difficulty": "easy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        yaml.safe_dump(
            {
                "provider": "fake",
                "currency": "CNY",
                "verified_at": "2026-08-21",
                "source_url": "https://example.com/pricing",
                "models": {
                    "fake-model": {"input_cache_miss": 1, "output": 2}
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / f"{mode}.yaml"
    payload = {
        "name": f"cli-{mode}",
        "mode": mode,
        "dataset_path": str(dataset),
        "knowledge_base_path": str(corpus),
        "database_path": str(tmp_path / "runs.sqlite3"),
        "artifact_dir": str(tmp_path / "reports"),
        "max_workers": 1,
        "provider": {
            "name": "fake",
            "base_url": "https://example.com/v1",
            "api_key_env": "FAKE_API_KEY",
            "chat_model": "fake-model",
            "embedding_model": "fake-embedding",
        },
        "retrieval": [
            {
                "chunk_size": 200,
                "chunk_overlap": 20,
                "top_k": 1,
                "prompt_variant": "direct",
            }
        ],
        "budget": {"currency": "CNY", "hard_limit": 20},
    }
    if mode == "live":
        payload["pricing_path"] = str(pricing)
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rag_quality_lab.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_offline_cli_generates_database_and_reports(tmp_path: Path) -> None:
    config = write_cli_fixture(tmp_path)

    result = run_cli("run", "--config", str(config))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert (tmp_path / "runs.sqlite3").exists()
    assert Path(payload["report_html"]).exists()
    assert Path(payload["report_json"]).exists()


def test_live_cli_requires_explicit_confirmation(tmp_path: Path) -> None:
    config = write_cli_fixture(tmp_path, mode="live")

    result = run_cli("run", "--config", str(config))

    assert result.returncode == 2
    assert "--confirm-live-run" in result.stderr


def test_legacy_mock_entry_delegates_to_new_cli(tmp_path: Path) -> None:
    config = write_cli_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "examples/run_eval.py",
            "--mock",
            "--config",
            str(config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "MOCK" in result.stdout
