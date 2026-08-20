"""Experiment orchestration primitives."""

from rag_quality_lab.experiments.budget import (
    BudgetExceeded,
    BudgetLedger,
    PlannedCall,
    PreflightDecision,
    calculate_actual_cost,
    estimate_tokens_upper_bound,
    preflight_budget,
)
from rag_quality_lab.experiments.compare import (
    ComparisonResult,
    RegressionConfig,
    RegressionRule,
    RegressionVerdict,
    compare_experiments,
    evaluate_regression,
)
from rag_quality_lab.experiments.runner import ProviderBundle, run_experiment
from rag_quality_lab.experiments.store import ExperimentStore

__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "ComparisonResult",
    "ExperimentStore",
    "ProviderBundle",
    "RegressionConfig",
    "RegressionRule",
    "RegressionVerdict",
    "PlannedCall",
    "PreflightDecision",
    "calculate_actual_cost",
    "compare_experiments",
    "estimate_tokens_upper_bound",
    "evaluate_regression",
    "preflight_budget",
    "run_experiment",
]
