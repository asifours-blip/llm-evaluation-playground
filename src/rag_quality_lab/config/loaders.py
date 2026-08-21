"""Load validated inputs from disk."""

import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from rag_quality_lab.domain.models import (
    Document,
    EvaluationDataset,
    ExperimentConfig,
    PricingConfig,
)

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
    updates = {
        field: _resolve_from(config_path, getattr(experiment, field))
        for field in (
            "dataset_path",
            "knowledge_base_path",
            "database_path",
            "artifact_dir",
        )
    }
    if experiment.pricing_path is not None:
        pricing_path = _resolve_from(config_path, experiment.pricing_path)
        pricing = load_yaml_model(pricing_path, PricingConfig)
        if experiment.budget.currency != pricing.currency:
            raise ValueError("budget and pricing currency must match")
        updates["pricing_path"] = pricing_path
    return experiment.model_copy(update=updates)


def validate_dataset_corpus(
    dataset: EvaluationDataset, documents: list[Document]
) -> None:
    """Validate document references and evidence against the loaded corpus."""

    documents_by_id = {document.id: document for document in documents}
    for case in dataset.cases:
        missing = sorted(set(case.expected_document_ids) - set(documents_by_id))
        if missing:
            raise ValueError(
                f"case {case.id} references unknown documents: {', '.join(missing)}"
            )
        expected_text = "\n".join(
            documents_by_id[document_id].text
            for document_id in case.expected_document_ids
        )
        if any(
            evidence not in expected_text for evidence in case.reference_evidence
        ):
            raise ValueError(f"case {case.id} evidence is absent from expected documents")


def _resolve_from(config_path: Path, value: Path) -> Path:
    if value.is_absolute():
        return value
    return (config_path.resolve().parent / value).resolve()
