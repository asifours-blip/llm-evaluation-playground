import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.metrics.calibration import AnnotationSnapshot, HumanAnnotation


def write_cli_fixture(
    tmp_path: Path, *, mode: str = "mock", with_judge: bool = False
) -> Path:
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
    if with_judge:
        payload["provider"]["judge_model"] = "fake-model"
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


def test_live_preflight_includes_generation_and_judge_calls(tmp_path: Path) -> None:
    config = write_cli_fixture(tmp_path, mode="live", with_judge=True)

    result = run_cli("run", "--config", str(config), "--preflight-only")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["unbuffered_cost"] == "0.031824"
    assert payload["buffered_cost"] == "0.03978000"
    assert payload["total_request_count"] == 12
    assert [call["output_token_cap"] for call in payload["planned_calls"]] == [
        3072,
        1536,
    ]
    assert [call["requests_per_case"] for call in payload["planned_calls"]] == [6, 6]
    assert payload["pricing_verified_at"] == "2026-08-21"
    assert payload["pricing_source_url"] == "https://example.com/pricing"


def test_confirmed_live_run_without_key_fails_without_traceback(
    tmp_path: Path,
) -> None:
    config = write_cli_fixture(tmp_path, mode="live")
    environment = os.environ.copy()
    environment.pop("FAKE_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_quality_lab.cli",
            "run",
            "--config",
            str(config),
            "--confirm-live-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "missing API key environment variable: FAKE_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


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


def test_judged_run_exports_blind_annotations_and_calibrates(tmp_path: Path) -> None:
    config = write_cli_fixture(tmp_path, with_judge=True)
    run_result = run_cli("run", "--config", str(config))
    run_payload = json.loads(run_result.stdout)
    annotations = tmp_path / "annotations.jsonl"

    export_result = run_cli(
        "annotate",
        "export",
        "--database",
        str(tmp_path / "runs.sqlite3"),
        "--experiment",
        run_payload["experiment_id"],
        "--count",
        "1",
        "--output",
        str(annotations),
    )

    assert export_result.returncode == 0, export_result.stderr
    blind = json.loads(annotations.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{24}", blind["sample_id"])
    assert "rag-001" not in blind["sample_id"]
    assert "chunk" not in blind["sample_id"]
    assert {"model", "config_id", "judge_score"}.isdisjoint(blind)
    assert re.fullmatch(r"[0-9a-f]{64}", blind["content_hash"])
    assert "ID: doc-01" in " ".join(blind["evidence"])

    blind["human_score"] = 5
    annotations.write_text(json.dumps(blind) + "\n", encoding="utf-8")
    import_result = run_cli(
        "annotate",
        "import",
        "--database",
        str(tmp_path / "runs.sqlite3"),
        "--experiment",
        run_payload["experiment_id"],
        "--input",
        str(annotations),
    )
    calibration = run_cli(
        "calibrate",
        "--database",
        str(tmp_path / "runs.sqlite3"),
        "--experiment",
        run_payload["experiment_id"],
    )

    assert import_result.returncode == 0, import_result.stderr
    assert calibration.returncode == 0, calibration.stderr
    calibration_payload = json.loads(calibration.stdout)
    assert calibration_payload["label_count"] == 1
    assert not calibration_payload["blocking_eligible"]

    report_result = run_cli(
        "report",
        "--database",
        str(tmp_path / "runs.sqlite3"),
        "--experiment",
        run_payload["experiment_id"],
        "--output",
        str(tmp_path / "calibrated-report"),
    )
    report_payload = json.loads(report_result.stdout)
    stored_report = json.loads(
        Path(report_payload["report_json"]).read_text(encoding="utf-8")
    )

    assert stored_report["judge_calibration"]["label_count"] == 1


def test_pairwise_cli_executes_both_orders_and_persists_report(tmp_path: Path) -> None:
    config = write_cli_fixture(tmp_path, with_judge=True)
    run_result = run_cli("run", "--config", str(config))
    experiment_id = json.loads(run_result.stdout)["experiment_id"]
    config_id = "chunk200-overlap20-top1-direct"

    result = run_cli(
        "pairwise",
        "--config",
        str(config),
        "--database",
        str(tmp_path / "runs.sqlite3"),
        "--baseline",
        experiment_id,
        "--candidate",
        experiment_id,
        "--baseline-config",
        config_id,
        "--candidate-config",
        config_id,
        "--output",
        str(tmp_path / "pairwise"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert report["summary"]["completed_count"] == 1.0
    assert report["summary"]["tie_rate"] == 1.0
    assert report["outcomes"][0]["forward"] is not None
    assert report["outcomes"][0]["reversed_order"] is not None


def test_database_regression_uses_stored_eligible_judge_calibration(
    tmp_path: Path,
) -> None:
    config = write_cli_fixture(tmp_path, with_judge=True)
    run_result = run_cli("run", "--config", str(config))
    experiment_id = json.loads(run_result.stdout)["experiment_id"]
    snapshots = [
        AnnotationSnapshot(
            sample_id=f"sample-{index}",
            source_case_id="rag-001",
            config_id="chunk200-overlap20-top1-direct",
            model="fake-model",
            judge_score=5,
            content_hash=f"{index:064x}",
        )
        for index in range(12)
    ]
    annotations = [
        HumanAnnotation(
            sample_id=snapshot.sample_id,
            human_score=5,
            content_hash=snapshot.content_hash,
        )
        for snapshot in snapshots
    ]
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        store.record_annotation_snapshots(experiment_id, snapshots)
        store.record_human_annotations(experiment_id, annotations)
    rules = tmp_path / "judge-rules.yaml"
    rules.write_text(
        yaml.safe_dump(
            {"rules": [{"metric": "judge_mean_score", "minimum_delta": 0.1}]}
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "regression",
        "--database",
        str(tmp_path / "runs.sqlite3"),
        "--baseline",
        experiment_id,
        "--candidate",
        experiment_id,
        "--rules",
        str(rules),
    )

    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["failed_metrics"] == ["judge_mean_score"]
    assert payload["skipped_metrics"] == []
