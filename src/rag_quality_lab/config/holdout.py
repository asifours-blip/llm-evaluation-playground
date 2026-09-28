"""Freeze and verify the holdout split of a versioned evaluation dataset.

A freeze records, per dataset name and version, the hash of the complete
content of every holdout case in a lock file next to the dataset
(``<dataset>.holdout-lock.json``). Once a version is frozen, any change to its
holdout (an edited case, a case moved into or out of the holdout) makes
verification fail, and runs refuse the dataset. The only way to change a
frozen holdout is to create a new dataset version, which is unfrozen until it
is frozen in turn. Dev cases may change within a version without affecting
the freeze. The lock file is meant to be committed; its Git history is the
audit trail of every freeze.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag_quality_lab.domain.models import UNLABELED, EvaluationDataset

HoldoutState = Literal["frozen", "not_frozen", "no_holdout"]


class HoldoutTampered(ValueError):
    """A frozen holdout changed without a new dataset version."""


class HoldoutFreeze(BaseModel):
    """The recorded holdout of one dataset version."""

    dataset_name: str
    dataset_version: str
    holdout_hash: str
    holdout_case_ids: list[str]
    frozen_at: datetime


class HoldoutLock(BaseModel):
    """Every holdout freeze recorded for a dataset file."""

    freezes: list[HoldoutFreeze] = Field(default_factory=list)

    def find(self, dataset: EvaluationDataset) -> HoldoutFreeze | None:
        return next(
            (
                freeze
                for freeze in self.freezes
                if freeze.dataset_name == dataset.name
                and freeze.dataset_version == dataset.version
            ),
            None,
        )


class HoldoutStatus(BaseModel):
    """Verified holdout state of one loaded dataset."""

    dataset_name: str
    dataset_version: str
    state: HoldoutState
    holdout_hash: str
    holdout_case_count: int
    frozen_at: datetime | None = None


def holdout_lock_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path.with_name(f"{path.stem}.holdout-lock.json")


def load_holdout_lock(path: str | Path) -> HoldoutLock:
    lock_path = Path(path)
    if not lock_path.exists():
        return HoldoutLock()
    return HoldoutLock.model_validate_json(lock_path.read_text(encoding="utf-8-sig"))


def verify_holdout(dataset: EvaluationDataset, lock: HoldoutLock) -> HoldoutStatus:
    """Return the holdout state, refusing a frozen holdout that has changed."""

    current = dataset.holdout_hash()
    holdout_ids = dataset.split_case_ids("holdout")
    freeze = lock.find(dataset)
    if freeze is not None and freeze.holdout_hash != current:
        added = sorted(set(holdout_ids) - set(freeze.holdout_case_ids))
        removed = sorted(set(freeze.holdout_case_ids) - set(holdout_ids))
        details = []
        if added:
            details.append("added: " + ", ".join(added))
        if removed:
            details.append("removed: " + ", ".join(removed))
        if not details:
            details.append("case content edited")
        raise HoldoutTampered(
            f"holdout of dataset {dataset.name} {dataset.version} was frozen with hash "
            f"{freeze.holdout_hash} at {freeze.frozen_at.isoformat()} but now hashes to "
            f"{current} ({'; '.join(details)}); a frozen holdout cannot change, so "
            "restore it or create a new dataset version"
        )
    if freeze is not None:
        state: HoldoutState = "frozen"
    elif holdout_ids:
        state = "not_frozen"
    else:
        state = "no_holdout"
    return HoldoutStatus(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        state=state,
        holdout_hash=current,
        holdout_case_count=len(holdout_ids),
        frozen_at=freeze.frozen_at if freeze is not None else None,
    )


def freeze_holdout(
    dataset: EvaluationDataset,
    lock_path: str | Path,
    *,
    now: datetime | None = None,
) -> HoldoutFreeze:
    """Record the holdout hash of this dataset version; idempotent for equal content."""

    holdout_ids = dataset.split_case_ids("holdout")
    if not holdout_ids:
        raise ValueError(
            f"dataset {dataset.name} {dataset.version} has no holdout cases to freeze"
        )
    lock = load_holdout_lock(lock_path)
    existing = lock.find(dataset)
    if existing is not None:
        if existing.holdout_hash != dataset.holdout_hash():
            raise HoldoutTampered(
                f"dataset {dataset.name} {dataset.version} is already frozen with a "
                "different holdout; create a new dataset version to freeze new content"
            )
        return existing
    freeze = HoldoutFreeze(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        holdout_hash=dataset.holdout_hash(),
        holdout_case_ids=holdout_ids,
        frozen_at=now or datetime.now(UTC),
    )
    lock.freezes.append(freeze)
    path = Path(lock_path)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        lock.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(path)
    return freeze


def split_summary(dataset: EvaluationDataset, lock: HoldoutLock) -> dict[str, object]:
    """Count cases per split and label, marking absent labels as unlabeled."""

    counts: dict[str, dict[str, int]] = {
        "splits": {},
        "difficulty": {},
        "review_status": {},
        "answerability": {},
    }
    for case in dataset.cases:
        labels = case.label_values()
        for key, label in (
            ("splits", labels["split"]),
            ("difficulty", labels["difficulty"]),
            ("review_status", labels["review_status"]),
            ("answerability", labels["answerability"]),
        ):
            counts[key][label] = counts[key].get(label, 0) + 1
    try:
        holdout: dict[str, object] = verify_holdout(dataset, lock).model_dump(mode="json")
    except HoldoutTampered as error:
        holdout = {"state": "tampered", "error": str(error)}
    return {
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "content_hash": dataset.content_hash(),
        "case_count": len(dataset.cases),
        "case_ids_by_split": {
            split: dataset.split_case_ids(split) for split in ("dev", "holdout")
        },
        "unlabeled_marker": UNLABELED,
        "holdout": holdout,
        **counts,
    }
