"""Command-line workflows for reproducible RAG evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from rag_quality_lab.config import load_dataset, load_experiment_config, load_yaml_model
from rag_quality_lab.domain.models import (
    Answerability,
    EvaluationDataset,
    ExperimentConfig,
    ExperimentStatus,
    PricingConfig,
    StructuredAnswer,
)
from rag_quality_lab.experiments import (
    PlannedCall,
    ProviderBundle,
    RegressionConfig,
    compare_experiments,
    evaluate_regression,
    preflight_budget,
    run_experiment,
)
from rag_quality_lab.experiments.runner import (
    GENERATION_INPUT_CAP,
    GENERATION_OUTPUT_CAP,
)
from rag_quality_lab.experiments.store import ExperimentStore
from rag_quality_lab.metrics.calibration import (
    HumanJudgePair,
    JudgeSample,
    calibrate,
    export_blind_annotations,
    import_human_annotations,
)
from rag_quality_lab.providers import (
    FakeChatProvider,
    FakeEmbeddingProvider,
    OpenAICompatibleProvider,
)
from rag_quality_lab.reporting import generate_reports

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

    report = subcommands.add_parser("report", help="regenerate a stored report")
    report.add_argument("--database", required=True, type=Path)
    report.add_argument("--experiment", required=True)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--badge", choices=("mock", "pilot", "final"))
    report.set_defaults(handler=_handle_report)

    compare = subcommands.add_parser("compare", help="compare two stored experiments")
    _add_comparison_arguments(compare)
    compare.set_defaults(handler=_handle_compare)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(CommandHandler, args.handler)
    try:
        return handler(args)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    return 2


def _handle_validate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    dataset = load_dataset(config.dataset_path)
    _print_json(
        {
            "status": "valid",
            "mode": config.mode,
            "case_count": len(dataset.cases),
            "arm_count": len(config.retrieval),
        }
    )
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
    _print_json(
        {
            "experiment_id": record.id,
            "status": record.status.value,
            "report_json": str(paths.json.resolve()),
            "report_html": str(paths.html.resolve()),
            "summary": record.summary,
        }
    )
    return 0 if record.status is ExperimentStatus.COMPLETED else 1


def _handle_report(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        record = store.get_experiment(experiment_id)
        badge = cast(Literal["mock", "pilot", "final"] | None, args.badge)
        paths = generate_reports(record, args.output, badge=badge)
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
    with ExperimentStore(args.database) as store:
        baseline = store.get_experiment(store.resolve_experiment_id(args.baseline))
        candidate = store.get_experiment(store.resolve_experiment_id(args.candidate))
    comparison = compare_experiments(baseline, candidate)
    _print_json(comparison.model_dump(mode="json"))
    return 0


def _handle_regression(args: argparse.Namespace) -> int:
    if args.fixture is not None:
        payload = json.loads(args.fixture.read_text(encoding="utf-8-sig"))
        passed = bool(payload.get("passed", False))
        _print_json(payload)
        return int(not passed)
    required = (args.database, args.baseline, args.candidate, args.rules)
    if any(value is None for value in required):
        raise ValueError(
            "regression requires --fixture or --database, --baseline, --candidate, and --rules"
        )
    with ExperimentStore(args.database) as store:
        baseline = store.get_experiment(store.resolve_experiment_id(args.baseline))
        candidate = store.get_experiment(store.resolve_experiment_id(args.candidate))
    rules = load_yaml_model(args.rules, RegressionConfig)
    verdict = evaluate_regression(
        compare_experiments(baseline, candidate), rules=rules.rules
    )
    _print_json(verdict.model_dump(mode="json"))
    return int(not verdict.passed)


def _handle_annotation_export(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        record = store.get_experiment(experiment_id)
    samples_by_case: dict[str, JudgeSample] = {}
    for result in record.case_results:
        if result.judge is None or result.answer is None:
            continue
        samples_by_case.setdefault(
            result.case_id,
            JudgeSample(
                case_id=result.case_id,
                question=result.question,
                reference_answer=result.reference_answer,
                candidate_answer=result.answer.answer,
                evidence=result.reference_evidence,
                model=result.model,
                config_id=result.config_id,
                judge_score=result.judge.score,
            ),
        )
    samples = [samples_by_case[key] for key in sorted(samples_by_case)][: args.count]
    if len(samples) < args.count:
        raise ValueError(
            f"experiment has {len(samples)} unique judge-scored cases; {args.count} requested"
        )
    export_blind_annotations(samples, args.output)
    _print_json(
        {"experiment_id": experiment_id, "count": len(samples), "output": str(args.output)}
    )
    return 0


def _handle_annotation_import(args: argparse.Namespace) -> int:
    annotations = import_human_annotations(args.input)
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        store.record_human_annotations(experiment_id, annotations)
    _print_json({"experiment_id": experiment_id, "count": len(annotations)})
    return 0


def _handle_calibrate(args: argparse.Namespace) -> int:
    with ExperimentStore(args.database) as store:
        experiment_id = store.resolve_experiment_id(args.experiment)
        record = store.get_experiment(experiment_id)
        annotations = store.get_human_annotations(experiment_id)
    judge_by_case = {
        result.case_id: result.judge.score
        for result in record.case_results
        if result.judge is not None
    }
    missing = sorted(
        annotation.case_id
        for annotation in annotations
        if annotation.case_id not in judge_by_case
    )
    if missing:
        raise ValueError(f"human labels have no judge score: {', '.join(missing)}")
    result = calibrate(
        [
            HumanJudgePair(
                human_score=annotation.human_score,
                judge_score=judge_by_case[annotation.case_id],
            )
            for annotation in annotations
        ]
    )
    _print_json(result.model_dump(mode="json"))
    return 0


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
        )
    chat = OpenAICompatibleProvider(
        base_url=str(config.provider.base_url),
        api_key_env=config.provider.api_key_env,
        timeout_seconds=config.provider.timeout_seconds,
        max_retries=config.provider.max_retries,
        extra_body=_provider_extra_body(config),
    )
    embedding = (
        FakeEmbeddingProvider(_fake_dimensions(config))
        if config.provider.embedding_model.startswith("fake-hash")
        else chat
    )
    return ProviderBundle(embedding=embedding, chat=chat)


def _live_preflight(config: ExperimentConfig, dataset: EvaluationDataset) -> Any:
    if config.pricing_path is None:
        raise ValueError("live experiments require a pricing file")
    pricing = load_yaml_model(config.pricing_path, PricingConfig)
    return preflight_budget(
        planned=[
            PlannedCall(
                model=config.provider.chat_model,
                input_token_cap=GENERATION_INPUT_CAP,
                output_token_cap=GENERATION_OUTPUT_CAP,
                count=len(dataset.cases) * len(config.retrieval),
            )
        ],
        pricing=pricing,
        budget=config.budget,
    )


def _provider_extra_body(config: ExperimentConfig) -> dict[str, Any]:
    if config.provider.name.casefold() == "deepseek":
        return {"thinking": {"type": "disabled"}}
    return {}


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


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
