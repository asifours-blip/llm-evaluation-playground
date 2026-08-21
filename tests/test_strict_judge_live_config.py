from pathlib import Path

from rag_quality_lab.config import load_dataset, load_experiment_config, load_yaml_model
from rag_quality_lab.domain.models import PricingConfig
from rag_quality_lab.experiments import preflight_budget
from rag_quality_lab.experiments.runner import planned_calls

ROOT = Path(__file__).resolve().parents[1]


def test_strict_judge_live_configuration_stays_under_budget() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "live-deepseek-flash-strict-judge.example.yaml"
    )
    dataset = load_dataset(config.dataset_path)
    assert config.pricing_path is not None
    pricing = load_yaml_model(config.pricing_path, PricingConfig)
    decision = preflight_budget(
        planned=planned_calls(config, len(dataset.cases) * len(config.retrieval)),
        pricing=pricing,
        budget=config.budget,
    )

    assert decision.allowed is True
    assert decision.buffered_cost < config.budget.hard_limit
