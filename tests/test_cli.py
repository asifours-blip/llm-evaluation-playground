import json
import subprocess
import sys
from pathlib import Path

import yaml


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
    assert payload["unbuffered_cost"] == "0.006036"
    assert payload["buffered_cost"] == "0.00754500"
    assert payload["total_request_count"] == 2
    assert [call["output_token_cap"] for call in payload["planned_calls"]] == [
        512,
        256,
    ]
    assert payload["pricing_verified_at"] == "2026-08-21"
    assert payload["pricing_source_url"] == "https://example.com/pricing"


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
    assert blind["case_id"].startswith("rag-001::chunk")
    assert {"model", "config_id", "judge_score"}.isdisjoint(blind)
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
