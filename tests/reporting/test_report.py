import hashlib
import json
from decimal import Decimal
from pathlib import Path

from rag_quality_lab.domain.models import (
    CaseResult,
    Chunk,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
    RetrievalHit,
    StructuredAnswer,
    TokenUsage,
)
from rag_quality_lab.reporting.report import generate_reports


def example_experiment() -> ExperimentRecord:
    return ExperimentRecord(
        id="experiment-001",
        identity=ExperimentIdentity(
            name="offline",
            mode="mock",
            commit_sha="abc123",
            dirty=False,
            dataset_hash="dataset-hash",
            prompt_hashes={"direct": "prompt-hash"},
            config={"retrieval": [{"top_k": 3}]},
            random_seed=42,
            python_version="3.11.9",
        ),
        status=ExperimentStatus.COMPLETED,
        case_results=[
            CaseResult(
                case_id="rag-001",
                category="retrieval",
                config_id="config-a",
                model="fake-model",
                answer=StructuredAnswer(
                    answer="RAG retrieves evidence.",
                    citations=["doc-01"],
                    abstained=False,
                ),
                retrieval_hits=[
                    RetrievalHit(
                        chunk=Chunk(
                            id="doc-01#chunk-000",
                            document_id="doc-01",
                            text="RAG retrieves evidence.",
                        ),
                        score=0.9,
                    )
                ],
                metrics={"answer_f1": 1.0, "retrieval_recall_at_k": 1.0},
                usage=TokenUsage(input_tokens=10, output_tokens=4),
                latency_ms=12.5,
                cost=Decimal("0.0001"),
                status="completed",
            )
        ],
        summary={
            "false_answer_rate": 0.0,
            "failure_count": 0.0,
            "answer_f1": 1.0,
        },
    )


def test_report_exposes_identity_quality_cost_and_failures(tmp_path: Path) -> None:
    paths = generate_reports(example_experiment(), tmp_path)

    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    html = paths.html.read_text(encoding="utf-8")

    assert payload["identity"]["dataset_hash"] == "dataset-hash"
    assert payload["summary"]["false_answer_rate"] == 0.0
    assert payload["system"]["p95_latency_ms"] == 12.5
    assert payload["system"]["total_tokens"] == 14
    assert payload["category_breakdown"]["retrieval"]["answer_f1"] == 1.0
    assert "P95 latency" in html
    assert "MOCK" in html
    assert "Retrieved evidence" in html
    assert "doc-01#chunk-000" in html
    assert "http://" not in html and "https://" not in html
    assert all(line == line.rstrip() for line in html.splitlines())
    assert paths.json_sha256 == hashlib.sha256(paths.json.read_bytes()).hexdigest()
    assert paths.html_sha256 == hashlib.sha256(paths.html.read_bytes()).hexdigest()


def test_live_report_defaults_to_pilot_not_final(tmp_path: Path) -> None:
    experiment = example_experiment().model_copy(
        update={
            "identity": example_experiment().identity.model_copy(
                update={"mode": "live"}
            )
        }
    )

    paths = generate_reports(experiment, tmp_path)
    payload = json.loads(paths.json.read_text(encoding="utf-8"))

    assert payload["badge"] == "pilot"
