"""Budgeted, two-order pairwise judge workflow."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from rag_quality_lab.domain.models import CaseResult, ExperimentConfig, ExperimentRecord
from rag_quality_lab.experiments.budget import BudgetLedger, PlannedCall
from rag_quality_lab.metrics.judge import (
    PairwiseCaseResult,
    PairwiseComparisonRecord,
    resolve_pairwise,
)
from rag_quality_lab.providers.base import JudgeProvider
from rag_quality_lab.providers.openai_compatible import (
    JUDGE_INPUT_TOKEN_CAP,
    JUDGE_OUTPUT_TOKEN_CAP,
    REPAIR_PROMPT_TOKEN_ALLOWANCE,
    ProviderError,
)


def planned_pairwise_calls(
    config: ExperimentConfig, pair_count: int
) -> list[PlannedCall]:
    """Reserve both answer orders, repairs, and every configured HTTP retry."""

    attempts = config.provider.max_retries + 1
    return [
        PlannedCall(
            model=_judge_model(config),
            input_token_cap=(
                JUDGE_INPUT_TOKEN_CAP
                + JUDGE_OUTPUT_TOKEN_CAP
                + REPAIR_PROMPT_TOKEN_ALLOWANCE
            )
            * attempts,
            output_token_cap=JUDGE_OUTPUT_TOKEN_CAP * 2 * attempts,
            count=pair_count,
            phase=phase,
            requests_per_case=2 * attempts,
        )
        for phase in ("pairwise_forward", "pairwise_reversed")
    ]


def run_pairwise_comparison(
    *,
    baseline: ExperimentRecord,
    candidate: ExperimentRecord,
    baseline_config_id: str,
    candidate_config_id: str,
    config: ExperimentConfig,
    judge: JudgeProvider,
    ledger: BudgetLedger | None = None,
) -> PairwiseComparisonRecord:
    """Compare matching case/model outputs in A/B and B/A order."""

    if baseline.identity.dataset_hash != candidate.identity.dataset_hash:
        raise ValueError("pairwise experiments must use the same dataset hash")
    baseline_cases = _selected_results(baseline, baseline_config_id)
    candidate_cases = _selected_results(candidate, candidate_config_id)
    keys = sorted(set(baseline_cases) & set(candidate_cases))
    if not keys:
        raise ValueError("pairwise configurations have no completed case/model pairs")
    if set(baseline_cases) != set(candidate_cases):
        raise ValueError("pairwise configurations must contain identical case/model keys")

    outcomes: list[PairwiseCaseResult] = []
    for key in keys:
        left = baseline_cases[key]
        right = candidate_cases[key]
        reservations: list[Decimal] = []
        if ledger is not None:
            reservations = ledger.reserve_many(planned_pairwise_calls(config, 1))
        outstanding = list(reservations)
        cost = Decimal("0")
        forward = None
        reversed_order = None
        try:
            evidence = _combined_evidence(left, right)
            forward = judge.pairwise(
                left.question,
                left.reference_answer,
                evidence,
                _answer(left),
                _answer(right),
                model=_judge_model(config),
            )
            if ledger is not None:
                cost += ledger.settle(
                    outstanding[0],
                    model=_judge_model(config),
                    usage=forward.usage,
                )
                outstanding.pop(0)
            reversed_order = judge.pairwise(
                left.question,
                left.reference_answer,
                evidence,
                _answer(right),
                _answer(left),
                model=_judge_model(config),
            )
            if ledger is not None:
                cost += ledger.settle(
                    outstanding[0],
                    model=_judge_model(config),
                    usage=reversed_order.usage,
                )
                outstanding.pop(0)
            resolved = resolve_pairwise(
                forward=forward.parsed,
                reversed_order=reversed_order.parsed,
            )
            outcomes.append(
                PairwiseCaseResult(
                    case_id=key[0],
                    model=key[1],
                    status="completed",
                    winner=_winner(resolved.winner),
                    position_sensitive=resolved.position_sensitive,
                    forward=resolved.forward,
                    reversed_order=resolved.reversed_order,
                    forward_usage=forward.usage,
                    reversed_usage=reversed_order.usage,
                    cost=cost,
                )
            )
        except Exception as error:
            estimated = Decimal("0")
            if ledger is not None and outstanding:
                if forward is None and len(outstanding) > 1:
                    estimated = ledger.charge_reserved(outstanding[:1])
                    ledger.release_reserved(outstanding[1:])
                else:
                    estimated = ledger.charge_reserved(outstanding)
            outcomes.append(
                PairwiseCaseResult(
                    case_id=key[0],
                    model=key[1],
                    status="failed",
                    forward=forward.parsed if forward is not None else None,
                    reversed_order=(
                        reversed_order.parsed if reversed_order is not None else None
                    ),
                    forward_usage=forward.usage if forward is not None else None,
                    reversed_usage=(
                        reversed_order.usage if reversed_order is not None else None
                    ),
                    cost=cost + estimated,
                    cost_estimated=bool(estimated),
                    error=(
                        str(error)
                        if isinstance(error, ProviderError)
                        else f"{type(error).__name__}: pairwise execution failed"
                    ),
                )
            )
    return PairwiseComparisonRecord(
        id=str(uuid.uuid4()),
        baseline_experiment_id=baseline.id,
        candidate_experiment_id=candidate.id,
        baseline_config_id=baseline_config_id,
        candidate_config_id=candidate_config_id,
        judge_model=_judge_model(config),
        outcomes=outcomes,
        summary=_summary(outcomes),
    )


def _selected_results(
    experiment: ExperimentRecord, config_id: str
) -> dict[tuple[str, str], CaseResult]:
    return {
        (result.case_id, result.model): result
        for result in experiment.case_results
        if result.config_id == config_id
        and result.status == "completed"
        and result.answer is not None
    }


def _answer(result: CaseResult) -> str:
    if result.answer is None:
        raise ValueError("pairwise result is missing an answer")
    return result.answer.answer


def _combined_evidence(left: CaseResult, right: CaseResult) -> list[str]:
    return list(
        dict.fromkeys(
            hit.chunk.text for result in (left, right) for hit in result.retrieval_hits
        )
    )


def _judge_model(config: ExperimentConfig) -> str:
    if config.provider.judge_model is None:
        raise ValueError("pairwise comparison requires provider.judge_model")
    return config.provider.judge_model


def _winner(
    preference: str | None,
) -> Literal["baseline", "candidate", "tie"] | None:
    return {"A": "baseline", "B": "candidate", "tie": "tie", None: None}[preference]


def _summary(outcomes: Sequence[PairwiseCaseResult]) -> dict[str, float]:
    completed = [outcome for outcome in outcomes if outcome.status == "completed"]
    stable = [outcome for outcome in completed if not outcome.position_sensitive]
    denominator = len(stable)
    return {
        "completed_count": float(len(completed)),
        "failure_count": float(len(outcomes) - len(completed)),
        "position_sensitive_count": float(
            sum(outcome.position_sensitive for outcome in completed)
        ),
        "baseline_win_rate": (
            sum(outcome.winner == "baseline" for outcome in stable) / denominator
            if denominator
            else 0.0
        ),
        "candidate_win_rate": (
            sum(outcome.winner == "candidate" for outcome in stable) / denominator
            if denominator
            else 0.0
        ),
        "tie_rate": (
            sum(outcome.winner == "tie" for outcome in stable) / denominator
            if denominator
            else 0.0
        ),
        "total_cost": float(sum((outcome.cost for outcome in outcomes), Decimal("0"))),
    }
