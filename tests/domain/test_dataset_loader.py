import json
from pathlib import Path

from rag_quality_lab.config.loaders import load_dataset


def test_load_dataset_validates_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    payload = {
        "version": "1.0.0",
        "name": "demo",
        "cases": [
            {
                "id": "rag-001",
                "question": "What is chunk overlap?",
                "reference_answer": "It repeats boundary text.",
                "answerability": "answerable",
                "expected_document_ids": ["doc-01"],
                "reference_evidence": ["Boundary text is repeated."],
                "category": "retrieval",
                "difficulty": "easy",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_dataset(path)

    assert dataset.name == "demo"
    assert dataset.cases[0].id == "rag-001"


def test_load_dataset_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    payload = {
        "version": "1.0.0",
        "name": "minimal",
        "cases": [
            {
                "id": "rag-001",
                "question": "What does RAG retrieve?",
                "reference_answer": "Evidence.",
                "answerability": "answerable",
                "expected_document_ids": ["doc-01"],
                "reference_evidence": ["Evidence."],
                "category": "retrieval",
                "difficulty": "easy",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8-sig")

    assert load_dataset(path).name == "minimal"
