"""Paired comparisons are gated on identical conditions and flag confounding."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_quality_lab.domain.models import (
    BudgetConfig,
    CaseResult,
    Chunk,
    ExperimentConfig,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
    ParameterDifference,
    ProviderConfig,
    RetrievalConfig,
    RetrievalHit,
    StructuredAnswer,
)
from rag_quality_lab.experiments.compare import (
    CONFOUNDED_WARNING,
    ComparisonRefused,
    compare_arms,
    compare_experiments,
)
from rag_quality_lab.experiments.pairwise import run_pairwise_comparison
from rag_quality_lab.providers.fake import FakeJudgeProvider
from rag_quality_lab.reporting.report import generate_comparison_report, generate_reports


def config_payload(
    arms: list[RetrievalConfig], *, seed: int = 42, name: str = "exp"
) -> dict[str, Any]:
    return ExperimentConfig(
        name=name,
        mode="mock",
        dataset_path=Path("dataset.json"),
        database_path=Path(f"{name}.sqlite3"),
        artifact_dir=Path(f"artifacts/{name}"),
        random_seed=seed,
        provider=ProviderConfig(
            name="fake",
            base_url="https://offline.invalid/v1",
            api_key_env="UNUSED",
            chat_model="fake-chat",
            embedding_model="fake-hash-64",
            judge_model="fake-judge",
        ),
        retrieval=arms,
        budget=BudgetConfig(hard_limit=Decimal("20")),
    ).model_dump(mode="json")


def arm(chunk_size: int = 300, top_k: int = 2, retriever: str = "embedding") -> RetrievalConfig:
    return RetrievalConfig.model_validate(
        {
            "chunk_size": chunk_size,
            "chunk_overlap": 50,
            "top_k": top_k,
            "prompt_variant": "direct",
            "retriever": retriever,
        }
    )


def hit(chunk_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk=Chunk(id=chunk_id, document_id=chunk_id.split("#")[0], text=f"text of {chunk_id}"),
        score=score,
    )


def result(
    case_id: str,
    config_id: str,
    hits: list[RetrievalHit],
    recall: float,
    split: str | None = "dev",
) -> CaseResult:
    return CaseResult.model_validate(
        {
            "case_id": case_id,
            "question": f"Question {case_id}?",
            "reference_answer": "Answer.",
            "config_id": config_id,
            "model": "fake-chat",
            "answer": StructuredAnswer(answer="Answer.", citations=[], abstained=False),
            "retrieval_hits": hits,
            "metrics": {"retrieval_recall_at_k": recall, "answer_f1": 1.0},
            "split": split,
            "status": "completed",
        }
    )


def record(
    experiment_id: str,
    retrieval: RetrievalConfig,
    results: list[CaseResult],
    **identity_overrides: Any,
) -> ExperimentRecord:
    identity: dict[str, Any] = {
        "name": experiment_id,
        "mode": "mock",
        "commit_sha": "abc123",
        "dirty": False,
        "dataset_hash": "dataset-hash",
        "dataset_version": "1.0.0",
        "prompt_hashes": {"direct": "prompt"},
        "config": config_payload([retrieval], name=experiment_id),
        "random_seed": 42,
        "python_version": "3.11",
        "corpus_hash": "corpus-hash",
        "prompt_version": "v1",
    }
    identity.update(identity_overrides)
    return ExperimentRecord(
        id=experiment_id,
        identity=ExperimentIdentity.model_validate(identity),
        status=ExperimentStatus.COMPLETED,
        case_results=results,
        summary={"retrieval_recall_at_k": 0.5},
    )


def pair(
    candidate_arm: RetrievalConfig, **candidate_identity: Any
) -> tuple[ExperimentRecord, ExperimentRecord]:
    base_arm = arm()
    baseline = record(
        "baseline",
        base_arm,
        [
            result(
                "rag-001",
                base_arm.config_id,
                [hit("doc-01#c0", 0.9), hit("doc-02#c0", 0.5)],
                0.0,
            ),
            result(
                "rag-002", base_arm.config_id, [hit("doc-03#c0", 0.8)], 1.0, split="holdout"
            ),
        ],
    )
    candidate = record(
        "candidate",
        candidate_arm,
        [
            result(
                "rag-001",
                candidate_arm.config_id,
                [hit("doc-02#c0", 0.7), hit("doc-04#c0", 0.6)],
                1.0,
            ),
            result(
                "rag-002",
                candidate_arm.config_id,
                [hit("doc-03#c0", 0.8)],
                1.0,
                split="holdout",
            ),
        ],
        **candidate_identity,
    )
    return baseline, candidate


def test_single_variable_pair_shows_ranked_evidence_and_no_warning() -> None:
    baseline, candidate = pair(arm(chunk_size=600))

    comparison = compare_arms(
        baseline, candidate, arm().config_id, arm(chunk_size=600).config_id
    )

    assert comparison.config_differences == [
        ParameterDifference(path="retrieval.chunk_size", baseline=300, candidate=600)
    ]
    assert comparison.warning is None
    first = comparison.cases[0]
    assert first.case_id == "rag-001"
    assert [(item.rank, item.chunk_id) for item in first.baseline_hits] == [
        (1, "doc-01#c0"),
        (2, "doc-02#c0"),
    ]
    assert [(item.rank, item.chunk_id) for item in first.candidate_hits] == [
        (1, "doc-02#c0"),
        (2, "doc-04#c0"),
    ]
    assert first.baseline_hits[0].text == "text of doc-01#c0"
    assert first.baseline_only_chunk_ids == ["doc-01#c0"]
    assert first.candidate_only_chunk_ids == ["doc-04#c0"]
    assert [(change.chunk_id, change.baseline_rank, change.candidate_rank)
            for change in first.rank_changes] == [("doc-02#c0", 2, 1)]
    assert first.metric_deltas["retrieval_recall_at_k"].absolute == 1.0
    assert first.direction == "improved"
    assert comparison.cases[1].direction == "unchanged"
    # Dev and holdout deltas are reported apart, never pooled.
    assert set(comparison.split_deltas) == {"dev", "holdout"}
    assert comparison.split_deltas["dev"]["retrieval_recall_at_k"].absolute == 1.0
    assert comparison.split_deltas["holdout"]["retrieval_recall_at_k"].absolute == 0.0


def test_multi_variable_pair_lists_every_difference_and_warns() -> None:
    candidate_arm = arm(chunk_size=600, top_k=4, retriever="bm25")
    baseline, candidate = pair(candidate_arm)

    comparison = compare_arms(baseline, candidate, arm().config_id, candidate_arm.config_id)

    assert [difference.path for difference in comparison.config_differences] == [
        "retrieval.chunk_size",
        "retrieval.retriever",
        "retrieval.top_k",
    ]
    assert comparison.warning == CONFOUNDED_WARNING
    assert CONFOUNDED_WARNING == "本次比较混杂多个变量，差异不能归因于单一改动"


def test_code_version_and_prompt_changes_count_as_differences() -> None:
    baseline, candidate = pair(arm(chunk_size=600), commit_sha="def456")

    comparison = compare_arms(
        baseline, candidate, arm().config_id, arm(chunk_size=600).config_id
    )

    assert [difference.path for difference in comparison.config_differences] == [
        "code_version.commit_sha",
        "retrieval.chunk_size",
    ]
    assert comparison.warning == CONFOUNDED_WARNING


@pytest.mark.parametrize(
    ("overrides", "drop_case", "expected"),
    [
        ({"dataset_version": "1.1.0", "dataset_hash": "other"}, False, "dataset version"),
        ({}, True, "question set"),
        ({"random_seed": 7}, False, "random seed"),
        ({"corpus_hash": "other-corpus"}, False, "corpus"),
        ({"corpus_hash": None}, False, "corpus"),
        ({"dataset_hash": "edited"}, False, "dataset content"),
    ],
)
def test_pair_is_refused_when_conditions_differ(
    overrides: dict[str, Any], drop_case: bool, expected: str
) -> None:
    candidate_arm = arm(chunk_size=600)
    baseline, candidate = pair(candidate_arm, **overrides)
    if drop_case:
        candidate = candidate.model_copy(update={"case_results": candidate.case_results[:1]})

    with pytest.raises(ComparisonRefused, match=expected):
        compare_arms(baseline, candidate, arm().config_id, candidate_arm.config_id)
    with pytest.raises(ComparisonRefused, match=expected):
        run_pairwise_comparison(
            baseline=baseline,
            candidate=candidate,
            baseline_config_id=arm().config_id,
            candidate_config_id=candidate_arm.config_id,
            config=ExperimentConfig.model_validate(baseline.identity.config),
            judge=FakeJudgeProvider(),
        )


def test_refusal_names_every_differing_condition() -> None:
    baseline, candidate = pair(
        arm(chunk_size=600), random_seed=7, corpus_hash="other", dataset_version="2.0.0"
    )

    with pytest.raises(ComparisonRefused) as refused:
        compare_arms(baseline, candidate, arm().config_id, arm(chunk_size=600).config_id)

    message = str(refused.value)
    for item in ("corpus", "dataset version", "random seed"):
        assert item in message


def test_pairwise_judge_records_config_differences() -> None:
    candidate_arm = arm(chunk_size=600, top_k=4)
    baseline, candidate = pair(candidate_arm)

    outcome = run_pairwise_comparison(
        baseline=baseline,
        candidate=candidate,
        baseline_config_id=arm().config_id,
        candidate_config_id=candidate_arm.config_id,
        config=ExperimentConfig.model_validate(baseline.identity.config),
        judge=FakeJudgeProvider(),
    )

    assert [d.path for d in outcome.config_differences] == [
        "retrieval.chunk_size",
        "retrieval.top_k",
    ]
    assert outcome.warning == CONFOUNDED_WARNING


def test_experiment_comparison_lists_config_differences() -> None:
    baseline, _ = pair(arm())
    same_arm = record("candidate", arm(), baseline.case_results)
    shifted = record("candidate", arm(), baseline.case_results, random_seed=7)

    assert compare_experiments(baseline, same_arm).config_differences == []
    assert compare_experiments(baseline, same_arm).warning is None
    assert [d.path for d in compare_experiments(baseline, shifted).config_differences] == [
        "random_seed"
    ]


def test_comparison_report_puts_the_warning_at_the_top(tmp_path: Path) -> None:
    candidate_arm = arm(chunk_size=600, top_k=4)
    baseline, candidate = pair(candidate_arm)
    comparison = compare_arms(baseline, candidate, arm().config_id, candidate_arm.config_id)

    paths = generate_comparison_report(comparison, tmp_path)

    html = paths.html.read_text(encoding="utf-8")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["warning"] == CONFOUNDED_WARNING
    assert html.index(CONFOUNDED_WARNING) < html.index("retrieval.chunk_size")
    assert html.index(CONFOUNDED_WARNING) < html.index("rag-001")
    assert "doc-04#c0" in html


def test_experiment_report_shows_a_confounded_baseline_warning_first(tmp_path: Path) -> None:
    baseline, _ = pair(arm())
    shifted = record("candidate", arm(), baseline.case_results, random_seed=7, commit_sha="x")

    paths = generate_reports(shifted, tmp_path, comparison=compare_experiments(baseline, shifted))

    html = paths.html.read_text(encoding="utf-8")
    assert html.index(CONFOUNDED_WARNING) < html.index("Reproducibility identity")


def write_two_arm_config(tmp_path: Path, second_arm: dict[str, Any]) -> Path:
    corpus = tmp_path / "knowledge_base"
    corpus.mkdir(exist_ok=True)
    (corpus / "doc-01-rag.md").write_text(
        "# RAG\n\nID: doc-01\n\nRAG retrieves evidence. " * 5, encoding="utf-8"
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "name": "paired-cli",
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
                        "split": "dev",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "paired-cli",
                "mode": "mock",
                "dataset_path": str(dataset),
                "knowledge_base_path": str(corpus),
                "database_path": str(tmp_path / "runs.sqlite3"),
                "artifact_dir": str(tmp_path / "reports"),
                "max_workers": 1,
                "provider": {
                    "name": "fake",
                    "base_url": "https://offline.invalid/v1",
                    "api_key_env": "UNUSED",
                    "chat_model": "fake-chat",
                    "embedding_model": "fake-hash-16",
                },
                "retrieval": [
                    {"chunk_size": 60, "chunk_overlap": 10, "top_k": 1, "prompt_variant": "direct"},
                    second_arm,
                ],
                "budget": {"currency": "CNY", "hard_limit": 20},
            }
        ),
        encoding="utf-8",
    )
    return config


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rag_quality_lab.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("second_arm", "second_id", "warned"),
    [
        (
            {"chunk_size": 60, "chunk_overlap": 10, "top_k": 1, "prompt_variant": "direct",
             "retriever": "bm25"},
            "chunk60-overlap10-top1-direct-bm25",
            False,
        ),
        (
            {"chunk_size": 90, "chunk_overlap": 10, "top_k": 2, "prompt_variant": "direct"},
            "chunk90-overlap10-top2-direct",
            True,
        ),
    ],
)
def test_compare_cli_pairs_two_arms_and_writes_a_report(
    tmp_path: Path, second_arm: dict[str, Any], second_id: str, warned: bool
) -> None:
    config = write_two_arm_config(tmp_path, second_arm)
    experiment_id = json.loads(run_cli("run", "--config", str(config)).stdout)["experiment_id"]

    result = run_cli(
        "compare",
        "--database",
        str(tmp_path / "runs.sqlite3"),
        "--baseline",
        experiment_id,
        "--candidate",
        experiment_id,
        "--baseline-config",
        "chunk60-overlap10-top1-direct",
        "--candidate-config",
        second_id,
        "--output",
        str(tmp_path / "comparison"),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    report = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert (report["warning"] == CONFOUNDED_WARNING) is warned
    assert report["cases"][0]["case_id"] == "rag-001"
    assert report["cases"][0]["candidate_hits"]
    assert Path(payload["report_html"]).exists()


def test_unrecorded_prompt_version_is_not_a_difference() -> None:
    baseline, candidate = pair(arm(chunk_size=600), prompt_version=None)
    changed = pair(arm(chunk_size=600), prompt_version="v2")[1]

    assert [
        d.path
        for d in compare_arms(
            baseline, candidate, arm().config_id, arm(chunk_size=600).config_id
        ).config_differences
    ] == ["retrieval.chunk_size"]
    assert [
        d.path
        for d in compare_arms(
            baseline, changed, arm().config_id, arm(chunk_size=600).config_id
        ).config_differences
    ] == ["prompts.version", "retrieval.chunk_size"]
