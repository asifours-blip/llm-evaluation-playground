"""Load validated inputs from disk."""

import json
from pathlib import Path

from rag_quality_lab.domain.models import EvaluationDataset


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load a UTF-8 JSON dataset and validate its domain invariants."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return EvaluationDataset.model_validate(payload)
