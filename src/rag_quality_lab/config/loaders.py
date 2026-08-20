"""Load validated inputs from disk."""

import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from rag_quality_lab.domain.models import EvaluationDataset, ExperimentConfig, PricingConfig

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load a UTF-8 JSON dataset and validate its domain invariants."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return EvaluationDataset.model_validate(payload)


def load_yaml_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    """Load a YAML file into the requested Pydantic model."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    return model_type.model_validate(payload)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment and validate constraints spanning its pricing file."""

    config_path = Path(path)
    experiment = load_yaml_model(config_path, ExperimentConfig)
    if experiment.pricing_path is None:
        return experiment

    pricing_path = experiment.pricing_path
    if not pricing_path.is_absolute():
        pricing_path = config_path.parent / pricing_path
    pricing = load_yaml_model(pricing_path, PricingConfig)
    if experiment.budget.currency != pricing.currency:
        raise ValueError("budget and pricing currency must match")
    return experiment.model_copy(update={"pricing_path": pricing_path.resolve()})
