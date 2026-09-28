"""Command-line workflows for reproducible RAG evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from rag_quality_lab.config import (
    load_dataset,
    load_experiment_config,
    load_yaml_model,
    validate_dataset_corpus,
)
from rag_quality_lab.config.holdout import (
    HoldoutTampered,
    freeze_holdout,
    holdout_lock_path,
    load_holdout_lock,
    split_summary,
    verify_holdout,
)
from rag_quality_lab.domain.models import (
    Answerability,
    EvaluationDataset,
    ExperimentConfig,
    ExperimentRecord,
    ExperimentStatus,
    PricingConfig,
    StructuredAnswer,
)
from rag_quality_lab.experiments import (
    BudgetLedger,
    ExperimentPreflight,
    ProviderBundle,
    RegressionConfig,
    RegressionFixture,
    compare_experiments,
    evaluate_regression,
    planned_pairwise_calls,
    preflight_budget,
    run_experiment,
    run_pairwise_comparison,
)
from rag_quality_lab.experiments.compare import compare_arms, require_pairable
from rag_quality_lab.experiments.runner import planned_calls, resume_experiment
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.metrics.calibration import (
    AnnotationSnapshot,
    CalibrationResult,
    HumanJudgePair,
    JudgeSample,
    calibrate,
    export_blind_annotations,
    import_human_annotations,
)
from rag_quality_lab.providers import (
    FakeChatProvider,
    FakeEmbeddingProvider,
    FakeJudgeProvider,
    OpenAICompatibleProvider,
    ProviderError,
)
from rag_quality_lab.providers.fake import is_local_embedding_model
from rag_quality_lab.reporting import generate_reports
from rag_quality_lab.reporting.report import generate_comparison_report
from rag_quality_lab.retrieval.index import load_documents

CommandHandler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(prog="rag-quality")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate a run configuration")
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=_handle_validate)

    run = subcommands.add_parser("run", help="run an experiment and write reports")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--artifact-dir", type=Path)
    run.add_argument("--confirm-live-run", action="store_true")
    run.add_argument("--preflight-only", action="store_true")
    run.set_defaults(handler=_handle_run)

    resume = subcommands.add_parser(
        "resume", help="resume an interrupted experiment from its checkpoint"
    )
    resume.add_argument("--experiment", required=True)
    resume.add_argument("--config", required=True, type=Path)
    resume.add_argument("--artifact-dir", type=Path)
    resume.add_argument("--confirm-live-run", action="store_true")
    resume.add_argument(
        "--retry-unknown",
        action="store_true",
        help="re-send case arms whose outcome is unknown; they are charged again",
    )
    resume.set_defaults(handler=_handle_resume)

    cancel = subcommands.add_parser(
        "cancel", help="stop an experiment from claiming further cases"
    )
    cancel.add_argument("--experiment", required=True)
    cancel.add_argument("--database", required=True, type=Path)
    cancel.set_defaults(handler=_handle_cancel)

    status = subcommands.add_parser(
        "status", help="show lifecycle, progress, and spend of stored experiments"
    )
    status.add_argument("--experiment")
    status.add_argument("--database", required=True, type=Path)
    status.set_defaults(handler=_handle_status)

    report = subcommands.add_parser("report", help="regenerate a stored report")
    report.add_argument("--database", required=True, type=Path)
    report.add_argument("--experiment", required=True)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--badge", choices=("mock", "pilot", "final"))
    report.add_argument("--baseline")
    report.set_defaults(handler=_handle_report)

    compare = subcommands.add_parser(
        "compare",
        help="compare two stored experiments, or two configuration arms question by question",
    )
    _add_comparison_arguments(compare)
    compare.add_argument("--baseline-config")
    compare.add_argument("--candidate-config")
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=_handle_compare)

    dataset = subcommands.add_parser(
        "dataset", help="inspect dataset splits and freeze or verify the holdout"
    )
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    splits = dataset_commands.add_parser("splits", help="show splits, labels, and holdout state")
    splits.add_argument("--dataset", required=True, type=Path)
    splits.set_defaults(handler=_handle_dataset_splits)
    freeze = dataset_commands.add_parser(
        "freeze-holdout", help="record the holdout hash of this dataset version"
    )
    freeze.add_argument("--dataset", required=True, type=Path)
    freeze.set_defaults(handler=_handle_dataset_freeze)
    verify = dataset_commands.add_parser(
        "verify", help="check that a frozen holdout is unchanged"
    )
    verify.add_argument("--dataset", required=True, type=Path)
    verify.set_defaults(handler=_handle_dataset_verify)

    regression = subcommands.add_parser("regression", help="evaluate regression rules")
    regression.add_argument("--fixture", type=Path)
    regression.add_argument("--database", type=Path)
    regression.add_argument("--baseline")
    regression.add_argument("--candidate")
    regression.add_argument("--rules", type=Path)
    regression.set_defaults(handler=_handle_regression)

    annotate = subcommands.add_parser("annotate", help="manage blind human labels")
    annotate_commands = annotate.add_subparsers(dest="annotate_command", required=True)
    export = annotate_commands.add_parser("export", help="export blind annotation JSONL")
    export.add_argument("--database", required=True, type=Path)
    export.add_argument("--experiment", required=True)
    export.add_argument("--count", required=True, type=_positive_int)
    export.add_argument("--output", required=True, type=Path)
    export.set_defaults(handler=_handle_annotation_export)
    import_command = annotate_commands.add_parser(
        "import", help="import completed human annotation JSONL"
    )
    import_command.add_argument("--database", required=True, type=Path)
    import_command.add_argument("--experiment", required=True)
    import_command.add_argument("--input", required=True, type=Path)
    import_command.set_defaults(handler=_handle_annotation_import)

    calibration = subcommands.add_parser("calibrate", help="measure judge agreement")
    calibration.add_argument("--database", required=True, type=Path)
    calibration.add_argument("--experiment", required=True)
    calibration.set_defaults(handler=_handle_calibrate)

    pairwise = subcommands.add_parser(
        "pairwise", help="compare two configurations in both answer orders"
    )
    pairwise.add_argument("--config", required=True, type=Path)
    pairwise.add_argument("--database", required=True, type=Path)
    pairwise.add_argument("--baseline", required=True)
    pairwise.add_argument("--candidate", required=True)
    pairwise.add_argument("--baseline-config", required=True)
    pairwise.add_argument("--candidate-config", required=True)
    pairwise.add_argument("--output", required=True, type=Path)
    pairwise.add_argument("--confirm-live-run", action="store_true")
    pairwise.add_argument("--preflight-only", action="store_true")
    pairwise.set_defaults(handler=_handle_pairwise)

    serve = subcommands.add_parser(
        "serve", help="run the local workbench (dataset/config browsing, runs, reports)"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--allow-live",
        action="store_true",
        help="permit live (paid) runs started from the workbench; offline by default",
    )
    serve.set_defaults(handler=_handle_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(CommandHandler, args.handler)
    try:
        return handler(args)
    except (KeyError, OSError, ProviderError, ValueError) as error:
        parser.error(str(error))
    return 2


def _handle_validate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    dataset = load_dataset(config.dataset_path)
    validate_dataset_corpus(dataset, load_documents(config.knowledge_base_path))
    holdout = verify_holdout(dataset, load_holdout_lock(holdout_lock_path(config.dataset_path)))
    _print_json(
        {
            "status": "valid",
            "mode": config.mode,
            "case_count": len(dataset.select_splits(config.splits).cases),
            "arm_count": len(config.retrieval),
            "dataset_version": dataset.version,
            "holdout_state": holdout.state,
        }
    )
    return 0


def _handle_dataset_splits(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    _print_json(split_summary(dataset, load_holdout_lock(holdout_lock_path(args.dataset))))
    return 0


def _handle_dataset_freeze(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    lock_path = holdout_lock_path(args.dataset)
    freeze = freeze_holdout(dataset, lock_path)
    _print_json({**freeze.model_dump(mode="json"), "lock_path": str(lock_path)})
    return 0


def _handle_dataset_verify(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    try:
        status = verify_holdout(dataset, load_holdout_lock(holdout_lock_path(args.dataset)))
    except HoldoutTampered as error:
        _print_json(
            {
                "dataset_name": dataset.name,
                "dataset_version": dataset.version,
                "state": "tampered",
                "error": str(error),
            }
        )
        return 1
    _print_json(status.model_dump(mode="json"))
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if args.artifact_dir is not None:
        config = config.model_copy(update={"artifact_dir": args.artifact_dir})
    dataset = load_dataset(config.dataset_path)
    if config.mode == "live":
        decision = _live_preflight(config, dataset)
        if args.preflight_only:
            _print_json(decision.model_dump(mode="json"))
            return int(not decision.allowed)
        if not args.confirm_live_run:
            raise ValueError("live runs require --confirm-live-run")
        if not decision.allowed:
            _print_json(decision.model_dump(mode="json"))
            return 1
    elif args.preflight_only:
        raise ValueError("--preflight-only is only valid for live configurations")

    record = run_experiment(config, _provider_bundle(config, dataset), dataset)
    return _publish_run(config, record)


def _handle_resume(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if args.artifact_dir is not None:
        config = config.model_copy(update={"artifact_dir": args.artifact_dir})
    if config.mode == "live" and not args.confirm_live_run:
        raise ValueError("live resumes require --confirm-live-run")
    dataset = load_dataset(config.dataset_path)
    with ExperimentStore(config.database_path) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
    record = resume_experiment(
        experiment_id,
        config,
        _provider_bundle(config, dataset),
        dataset,
        retry_unknown=args.retry_unknown,
    )
    return _publish_run(config, record)


def _handle_cancel(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        status = store.request_cancel(experiment_id)
    _print_json(
        {
            "experiment_id": experiment_id,
            "status": status.value,
            "cancel_requested": True,
        }
    )
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        store.reap_orphans()
        if args.experiment is not None:
            payload = store.progress(store.resolve_experiment_id(args.experiment))
        else:
            payload = {
                "experiments": [
                    store.progress(experiment_id)
                    for experiment_id in store.experiment_ids()
                ]
            }
    _print_json(payload)
    return 0


def _publish_run(config: ExperimentConfig, record: ExperimentRecord) -> int:
    paths = generate_reports(record, config.artifact_dir)
    with ExperimentStore(config.database_path) as store:
        store.record_artifact(
            record.id,
            kind="json_report",
            path=str(paths.json),
            sha256=paths.json_sha256,
        )
        store.record_artifact(
            record.id,
            kind="html_report",
            path=str(paths.html),
            sha256=paths.html_sha256,
        )
    payload: dict[str, object] = {
        "experiment_id": record.id,
        "status": record.status.value,
        "report_json": str(paths.json.resolve()),
        "report_html": str(paths.html.resolve()),
        "summary": record.summary,
    }
    if record.unknown_calls:
        payload["unknown_outcome_task_keys"] = sorted(
            {call.task_key for call in record.unknown_calls}
        )
    _print_json(payload)
    return 0 if record.status is ExperimentStatus.COMPLETED else 1


def _handle_report(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        record = store.get_experiment(experiment_id)
        calibration = _calibration_for_record(store, experiment_id, record)
        comparison = None
        if args.baseline is not None:
            baseline = store.get_experiment(
                store.resolve_experiment_id(args.baseline)
            )
            comparison = compare_experiments(baseline, record)
        badge = cast(Literal["mock", "pilot", "final"] | None, args.badge)
        paths = generate_reports(
            record,
            args.output,
            badge=badge,
            calibration=calibration,
            comparison=comparison,
        )
        store.record_artifact(
            experiment_id,
            kind="json_report",
            path=str(paths.json),
            sha256=paths.json_sha256,
        )
        store.record_artifact(
            experiment_id,
            kind="html_report",
            path=str(paths.html),
            sha256=paths.html_sha256,
        )
    _print_json(_report_paths_payload(experiment_id, paths))
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    arms = (args.baseline_config, args.candidate_config)
    if any(arm is not None for arm in arms) and not all(arm is not None for arm in arms):
        raise ValueError("--baseline-config and --candidate-config must be given together")
    with ExperimentStore(args.database) as store:
        baseline = store.get_experiment(store.resolve_experiment_id(args.baseline))
        candidate = store.get_experiment(store.resolve_experiment_id(args.candidate))
    if args.baseline_config is None:
        if args.output is not None:
            raise ValueError("--output requires --baseline-config and --candidate-config")
        comparison = compare_experiments(baseline, candidate)
        _print_json(comparison.model_dump(mode="json"))
        return 0
    paired = compare_arms(baseline, candidate, args.baseline_config, args.candidate_config)
    if args.output is None:
        _print_json(paired.model_dump(mode="json"))
        return 0
    paths = generate_comparison_report(paired, args.output)
    _print_json(
        {
            "report_json": str(paths.json.resolve()),
            "report_html": str(paths.html.resolve()),
            "json_sha256": paths.json_sha256,
            "html_sha256": paths.html_sha256,
            "config_differences": [
                difference.model_dump(mode="json") for difference in paired.config_differences
            ],
            "warning": paired.warning,
        }
    )
    return 0


def _handle_regression(args: argparse.Namespace) -> int:
    if args.fixture is not None:
        fixture = RegressionFixture.model_validate_json(
            args.fixture.read_text(encoding="utf-8-sig")
        )
        fixture_dir = args.fixture.resolve().parent
        config_path = _fixture_path(fixture_dir, fixture.config_path)
        baseline_path = _fixture_path(fixture_dir, fixture.baseline_report_path)
        config = load_experiment_config(config_path)
        if config.mode != "mock":
            raise ValueError("regression fixtures must rerun a mock configuration")
        dataset = load_dataset(config.dataset_path)
        baseline = ExperimentRecord.model_validate_json(
            baseline_path.read_text(encoding="utf-8-sig")
        )
        with tempfile.TemporaryDirectory(prefix="rag-quality-regression-") as temp_dir:
            temporary_root = Path(temp_dir)
            candidate_config = config.model_copy(
                update={
                    "database_path": temporary_root / "experiments.sqlite3",
                    "artifact_dir": temporary_root / "artifacts",
                }
            )
            candidate = run_experiment(
                candidate_config,
                _provider_bundle(candidate_config, dataset),
                dataset,
            )
        verdict = evaluate_regression(
            compare_experiments(baseline, candidate),
            rules=fixture.rules,
        )
        _print_json(verdict.model_dump(mode="json"))
        return int(not verdict.passed)
    required = (args.database, args.baseline, args.candidate, args.rules)
    if any(value is None for value in required):
        raise ValueError(
            "regression requires --fixture or --database, --baseline, --candidate, and --rules"
        )
    with ExperimentStore(args.database) as store:
        baseline = store.get_experiment(store.resolve_experiment_id(args.baseline))
        candidate_id = store.resolve_experiment_id(args.candidate)
        candidate = store.get_experiment(candidate_id)
        judge_calibration = _calibration_for_record(store, candidate_id, candidate)
    rules = load_yaml_model(args.rules, RegressionConfig)
    verdict = evaluate_regression(
        compare_experiments(baseline, candidate),
        rules=rules.rules,
        judge_calibration=judge_calibration,
    )
    _print_json(verdict.model_dump(mode="json"))
    return int(not verdict.passed)


def _handle_annotation_export(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        record = store.get_experiment(experiment_id)
        samples = _stratified_judge_samples(record, experiment_id, args.count)
        blind_annotations = export_blind_annotations(samples, args.output)
        blind_by_id = {
            annotation.sample_id: annotation for annotation in blind_annotations
        }
        store.record_annotation_snapshots(
            experiment_id,
            [
                AnnotationSnapshot(
                    sample_id=sample.sample_id,
                    source_case_id=_source_case_id(
                        record, sample.sample_id, experiment_id
                    ),
                    config_id=sample.config_id,
                    model=sample.model,
                    judge_score=sample.judge_score,
                    content_hash=blind_by_id[sample.sample_id].content_hash,
                )
                for sample in samples
            ],
        )
    _print_json(
        {"experiment_id": experiment_id, "count": len(samples), "output": str(args.output)}
    )
    return 0


def _stratified_judge_samples(
    record: ExperimentRecord, experiment_id: str, count: int
) -> list[JudgeSample]:
    """Seeded round-robin sampling across evaluation and configuration strata."""

    candidates: dict[tuple[str, str, str, str, str], list[JudgeSample]] = {}
    for result in record.case_results:
        if result.judge is None or result.answer is None:
            continue
        sample_id = _opaque_sample_id(
            experiment_id, result.case_id, result.config_id, result.model
        )
        stratum: tuple[str, str, str, str, str] = (
            result.answerability.value if result.answerability is not None else "unknown",
            result.category or "unknown",
            result.difficulty or "unknown",
            result.config_id,
            result.model,
        )
        candidates.setdefault(stratum, []).append(
            JudgeSample(
                sample_id=sample_id,
                question=result.question,
                reference_answer=result.reference_answer,
                candidate_answer=result.answer.answer,
                evidence=[hit.chunk.text for hit in result.retrieval_hits],
                model=result.model,
                config_id=result.config_id,
                judge_score=result.judge.score,
            )
        )
    available = sum(len(samples) for samples in candidates.values())
    if available < count:
        raise ValueError(
            f"experiment has {available} unique judge-scored cases; {count} requested"
        )
    for stratum_samples in candidates.values():
        stratum_samples.sort(
            key=lambda sample: hashlib.sha256(
                f"{record.identity.random_seed}\0{sample.sample_id}".encode()
            ).hexdigest()
        )
    selected: list[JudgeSample] = []
    strata = sorted(candidates)
    while len(selected) < count:
        for stratum in strata:
            if candidates[stratum]:
                selected.append(candidates[stratum].pop(0))
                if len(selected) == count:
                    break
    return selected


def _opaque_sample_id(
    experiment_id: str, case_id: str, config_id: str, model: str
) -> str:
    value = f"annotation-v1\0{experiment_id}\0{case_id}\0{config_id}\0{model}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _source_case_id(
    record: ExperimentRecord, sample_id: str, experiment_id: str
) -> str:
    for result in record.case_results:
        if (
            _opaque_sample_id(
                experiment_id, result.case_id, result.config_id, result.model
            )
            == sample_id
        ):
            return result.case_id
    raise ValueError(f"opaque annotation sample has no source: {sample_id}")


def _handle_annotation_import(args: argparse.Namespace) -> int:
    annotations = import_human_annotations(args.input)
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        snapshots = {
            snapshot.sample_id: snapshot
            for snapshot in store.get_annotation_snapshots(experiment_id)
        }
        for annotation in annotations:
            snapshot = snapshots.get(annotation.sample_id)
            if snapshot is None:
                raise ValueError(
                    "annotation sample was not exported for this experiment: "
                    f"{annotation.sample_id}"
                )
            if annotation.content_hash != snapshot.content_hash:
                raise ValueError(
                    f"annotation snapshot hash mismatch: {annotation.sample_id}"
                )
        store.record_human_annotations(experiment_id, annotations)
    _print_json({"experiment_id": experiment_id, "count": len(annotations)})
    return 0


def _handle_calibrate(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        record = store.get_experiment(experiment_id)
        result = _calibration_for_record(store, experiment_id, record)
    if result is None:
        result = calibrate([])
    _print_json(result.model_dump(mode="json"))
    return 0


def _handle_pairwise(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if config.provider.judge_model is None:
        raise ValueError("pairwise comparison requires provider.judge_model")
    with ExperimentStore(args.database) as store:
        baseline_id = store.resolve_experiment_id(args.baseline)
        candidate_id = store.resolve_experiment_id(args.candidate)
        baseline = store.get_experiment(baseline_id)
        candidate = store.get_experiment(candidate_id)
        # Refuse mismatched conditions before planning or sending any request.
        require_pairable(baseline, candidate, args.baseline_config, args.candidate_config)
        pair_count = _pairwise_case_count(
            baseline,
            candidate,
            args.baseline_config,
            args.candidate_config,
        )
        pricing = None
        ledger = None
        if config.mode == "live":
            if config.pricing_path is None:
                raise ValueError("live pairwise comparison requires a pricing file")
            pricing = load_yaml_model(config.pricing_path, PricingConfig)
            plan = planned_pairwise_calls(config, pair_count)
            decision = preflight_budget(
                planned=plan,
                pricing=pricing,
                budget=config.budget,
            )
            preflight = ExperimentPreflight(
                **decision.model_dump(),
                planned_calls=plan,
                total_request_count=sum(
                    call.count * call.requests_per_case for call in plan
                ),
                pricing_provider=pricing.provider,
                pricing_verified_at=pricing.verified_at,
                pricing_source_url=str(pricing.source_url),
                pricing_rate_basis=pricing.rate_basis,
            )
            if args.preflight_only:
                _print_json(preflight.model_dump(mode="json"))
                return int(not preflight.allowed)
            if not args.confirm_live_run:
                raise ValueError("live pairwise comparisons require --confirm-live-run")
            if not preflight.allowed:
                _print_json(preflight.model_dump(mode="json"))
                return 1
            ledger = BudgetLedger(budget=config.budget, pricing=pricing)
        elif args.preflight_only:
            raise ValueError("--preflight-only is only valid for live configurations")

        dataset = load_dataset(config.dataset_path)
        judge = _provider_bundle(config, dataset).judge
        if judge is None:
            raise ValueError("pairwise comparison requires a judge provider")
        record = run_pairwise_comparison(
            baseline=baseline,
            candidate=candidate,
            baseline_config_id=args.baseline_config,
            candidate_config_id=args.candidate_config,
            config=config,
            judge=judge,
            ledger=ledger,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        report_path = args.output / f"pairwise-{record.id}.json"
        report_text = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        report_path.write_text(report_text, encoding="utf-8", newline="\n")
        report_sha256 = hashlib.sha256(report_text.encode()).hexdigest()
        store.record_pairwise_comparison(record)
        store.record_artifact(
            baseline_id,
            kind="pairwise_report",
            path=str(report_path),
            sha256=report_sha256,
            metadata={"candidate_experiment_id": candidate_id},
        )
    _print_json(
        {
            "comparison_id": record.id,
            "report_json": str(report_path.resolve()),
            "sha256": report_sha256,
            "summary": record.summary,
        }
    )
    return int(record.summary["failure_count"] > 0)


def _handle_serve(args: argparse.Namespace) -> int:
    from rag_quality_lab.web import serve

    server = serve(host=args.host, port=args.port, allow_live=args.allow_live)
    mode = "live runs allowed" if args.allow_live else "offline only"
    print(f"rag-quality workbench on http://{args.host}:{args.port} ({mode}); Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _pairwise_case_count(
    baseline: ExperimentRecord,
    candidate: ExperimentRecord,
    baseline_config_id: str,
    candidate_config_id: str,
) -> int:
    baseline_keys = {
        (result.case_id, result.model)
        for result in baseline.case_results
        if result.config_id == baseline_config_id
        and result.status == "completed"
        and result.answer is not None
    }
    candidate_keys = {
        (result.case_id, result.model)
        for result in candidate.case_results
        if result.config_id == candidate_config_id
        and result.status == "completed"
        and result.answer is not None
    }
    if not baseline_keys or baseline_keys != candidate_keys:
        raise ValueError("pairwise configurations must contain identical case/model keys")
    return len(baseline_keys)


def _calibration_for_record(
    store: ExperimentStore,
    experiment_id: str,
    record: ExperimentRecord,
) -> CalibrationResult | None:
    annotations = store.get_human_annotations(experiment_id)
    if not annotations:
        return None
    snapshots = {
        snapshot.sample_id: snapshot
        for snapshot in store.get_annotation_snapshots(experiment_id)
    }
    missing = sorted(
        annotation.sample_id
        for annotation in annotations
        if annotation.sample_id not in snapshots
        or annotation.content_hash != snapshots[annotation.sample_id].content_hash
    )
    if missing:
        raise ValueError(f"human labels have no judge score: {', '.join(missing)}")
    return calibrate(
        [
            HumanJudgePair(
                human_score=annotation.human_score,
                judge_score=snapshots[annotation.sample_id].judge_score,
            )
            for annotation in annotations
        ]
    )


def _provider_bundle(
    config: ExperimentConfig, dataset: EvaluationDataset
) -> ProviderBundle:
    if config.mode == "mock":
        answers = {
            case.question: StructuredAnswer(
                answer=case.reference_answer,
                citations=case.expected_document_ids,
                abstained=case.answerability is not Answerability.ANSWERABLE,
            )
            for case in dataset.cases
        }
        return ProviderBundle(
            embedding=FakeEmbeddingProvider(_fake_dimensions(config)),
            chat=FakeChatProvider(answers),
            judge=(
                FakeJudgeProvider()
                if config.provider.judge_model is not None
                else None
            ),
        )
    chat = OpenAICompatibleProvider(
        base_url=str(config.provider.base_url),
        api_key_env=config.provider.api_key_env,
        timeout_seconds=config.provider.timeout_seconds,
        max_retries=config.provider.max_retries,
        extra_body=_provider_extra_body(config),
        embedding_batch_size=config.provider.embedding_batch_size,
        embedding_batch_token_limit=config.provider.embedding_batch_token_limit,
    )
    embedding = (
        FakeEmbeddingProvider(_fake_dimensions(config))
        if is_local_embedding_model(config.provider.embedding_model)
        else chat
    )
    return ProviderBundle(
        embedding=embedding,
        chat=chat,
        judge=chat if config.provider.judge_model is not None else None,
    )


def _live_preflight(
    config: ExperimentConfig, dataset: EvaluationDataset
) -> ExperimentPreflight:
    if config.pricing_path is None:
        raise ValueError("live experiments require a pricing file")
    pricing = load_yaml_model(config.pricing_path, PricingConfig)
    plan = planned_calls(config, dataset, load_documents(config.knowledge_base_path))
    decision = preflight_budget(
        planned=plan,
        pricing=pricing,
        budget=config.budget,
    )
    return ExperimentPreflight(
        **decision.model_dump(),
        planned_calls=plan,
        total_request_count=sum(call.count * call.requests_per_case for call in plan),
        pricing_provider=pricing.provider,
        pricing_verified_at=pricing.verified_at,
        pricing_source_url=str(pricing.source_url),
        pricing_rate_basis=pricing.rate_basis,
    )


def _provider_extra_body(config: ExperimentConfig) -> dict[str, Any]:
    body: dict[str, Any] = {
        "temperature": config.provider.temperature,
        "top_p": config.provider.top_p,
    }
    if config.provider.name.casefold() == "deepseek":
        body["thinking"] = {"type": "disabled"}
    return body


def _fake_dimensions(config: ExperimentConfig) -> int:
    final = config.provider.embedding_model.rsplit("-", maxsplit=1)[-1]
    return int(final) if final.isdigit() else 64


def _add_comparison_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)


def _report_paths_payload(experiment_id: str, paths: Any) -> dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "report_json": str(paths.json.resolve()),
        "report_html": str(paths.html.resolve()),
        "json_sha256": paths.json_sha256,
        "html_sha256": paths.html_sha256,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _fixture_path(fixture_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (fixture_dir / path).resolve()


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
