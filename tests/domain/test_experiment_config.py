from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rag_quality_lab.config.loaders import load_experiment_config, load_yaml_model
from rag_quality_lab.domain.models import (
    BudgetConfig,
    ExperimentConfig,
    ModelPrice,
    PricingConfig,
    ProviderConfig,
    RetrievalConfig,
)


def provider_config() -> ProviderConfig:
    return ProviderConfig(
        name="fake",
        base_url="https://example.com/v1",
        api_key_env="FAKE_API_KEY",
        chat_model="fake-chat",
        embedding_model="fake-embedding",
    )


def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        chunk_size=400,
        chunk_overlap=50,
        top_k=3,
        prompt_variant="direct",
    )


def test_budget_requires_positive_hard_limit() -> None:
    with pytest.raises(ValidationError):
        BudgetConfig(currency="CNY", hard_limit=0)


def test_pricing_reports_staleness() -> None:
    pricing = PricingConfig(
        provider="deepseek",
        currency="CNY",
        verified_at=date.today() - timedelta(days=8),
        source_url="https://example.com/pricing",
        models={"flash": ModelPrice(input_cache_miss=1, output=2)},
    )

    assert pricing.is_stale(date.today(), max_age_days=7)


def test_retrieval_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(
            chunk_size=50,
            chunk_overlap=50,
            top_k=3,
            prompt_variant="direct",
        )


def test_experiment_rejects_duplicate_retrieval_configurations() -> None:
    retrieval = retrieval_config()
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="duplicate",
            mode="mock",
            dataset_path=Path("dataset.json"),
            database_path=Path("experiments.sqlite3"),
            artifact_dir=Path("artifacts"),
            provider=provider_config(),
            retrieval=[retrieval, retrieval],
            budget=BudgetConfig(hard_limit=20),
        )


def test_live_experiment_requires_pricing_path() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            name="live",
            mode="live",
            dataset_path=Path("dataset.json"),
            database_path=Path("experiments.sqlite3"),
            artifact_dir=Path("artifacts"),
            provider=provider_config(),
            retrieval=[retrieval_config()],
            budget=BudgetConfig(hard_limit=20),
        )


def test_yaml_loader_returns_typed_model(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "provider": "deepseek",
                "currency": "CNY",
                "verified_at": date.today().isoformat(),
                "source_url": "https://example.com/pricing",
                "models": {"flash": {"input_cache_miss": 1, "output": 2}},
            }
        ),
        encoding="utf-8",
    )

    pricing = load_yaml_model(path, PricingConfig)

    assert pricing.models["flash"].output == Decimal("2")


def test_experiment_loader_rejects_pricing_currency_mismatch(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        yaml.safe_dump(
            {
                "provider": "deepseek",
                "currency": "USD",
                "verified_at": date.today().isoformat(),
                "source_url": "https://example.com/pricing",
                "models": {"flash": {"input_cache_miss": 1, "output": 2}},
            }
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "name": "live",
                "mode": "live",
                "dataset_path": "dataset.json",
                "database_path": "experiments.sqlite3",
                "artifact_dir": "artifacts",
                "provider": provider_config().model_dump(mode="json"),
                "retrieval": [retrieval_config().model_dump(mode="json")],
                "budget": {"currency": "CNY", "hard_limit": 20},
                "pricing_path": pricing_path.name,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="currency"):
        load_experiment_config(experiment_path)
