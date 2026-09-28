"""Shared fixtures for workbench tests: a tiny mock-mode dataset and config."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

CASES = [
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
        ],
        start=1,
    )
]
CORPUS_TEXT = (
    "# RAG\n\nID: doc-01\n\nRAG retrieves evidence. Chunking splits documents. "
    "Reranking orders passages."
)


def write_mock_workbench_inputs(workdir: Path) -> Path:
    """Write a corpus, dataset, and mock-mode config; return the config path."""

    workdir.mkdir(parents=True, exist_ok=True)
    corpus = workdir / "knowledge_base"
    corpus.mkdir(exist_ok=True)
    (corpus / "doc-01-rag.md").write_text(CORPUS_TEXT, encoding="utf-8")
    dataset_path = workdir / "dataset.json"
    dataset_path.write_text(
        json.dumps({"version": "1.0.0", "name": "workbench-test", "cases": CASES}),
        encoding="utf-8",
    )
    config_path = workdir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "workbench-test",
                "mode": "mock",
                "dataset_path": str(dataset_path),
                "knowledge_base_path": str(corpus),
                "database_path": str(workdir / "experiments.sqlite3"),
                "artifact_dir": str(workdir / "artifacts"),
                "random_seed": 42,
                "max_workers": 2,
                "provider": {
                    "name": "fake",
                    "base_url": "https://offline.invalid/v1",
                    "api_key_env": "WORKBENCH_TEST_UNUSED_KEY",
                    "chat_model": "fake-chat",
                    "embedding_model": "fake-hash-64",
                },
                "retrieval": [
                    {
                        "chunk_size": 300,
                        "chunk_overlap": 50,
                        "top_k": 2,
                        "prompt_variant": "direct",
                    },
                    {
                        "chunk_size": 300,
                        "chunk_overlap": 50,
                        "top_k": 2,
                        "prompt_variant": "evidence_first",
                    },
                ],
                "budget": {
                    "currency": "CNY",
                    "hard_limit": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path
