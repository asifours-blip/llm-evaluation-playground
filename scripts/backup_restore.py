"""Back up and restore everything one experiment configuration touches.

A backup bundles the SQLite experiment database, the dataset (plus its
holdout lock, if frozen), the knowledge-base corpus, the pricing file (if
any), and every artifact under ``artifact_dir`` into one zip archive next to
a manifest that records where each piece came from. A restore extracts that
archive into a destination directory and writes a new config file whose
paths point at the restored copies, so the restored config, dataset, corpus,
and database are usable together without touching the originals.

This does not re-implement any experiment logic: it only copies files the
config already names (:class:`rag_quality_lab.domain.models.ExperimentConfig`)
plus the artifact directory recorded in it.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rag_quality_lab.config import load_experiment_config
from rag_quality_lab.config.holdout import holdout_lock_path

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Manifest:
    """Where each backed-up piece lives inside the archive."""

    created_at: str
    config_member: str
    database_member: str
    dataset_member: str
    holdout_lock_member: str | None
    knowledge_base_prefix: str
    pricing_member: str | None
    artifact_prefix: str

    def to_json(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "config_member": self.config_member,
            "database_member": self.database_member,
            "dataset_member": self.dataset_member,
            "holdout_lock_member": self.holdout_lock_member,
            "knowledge_base_prefix": self.knowledge_base_prefix,
            "pricing_member": self.pricing_member,
            "artifact_prefix": self.artifact_prefix,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> Manifest:
        return cls(
            created_at=str(payload["created_at"]),
            config_member=str(payload["config_member"]),
            database_member=str(payload["database_member"]),
            dataset_member=str(payload["dataset_member"]),
            holdout_lock_member=(
                str(payload["holdout_lock_member"])
                if payload.get("holdout_lock_member") is not None
                else None
            ),
            knowledge_base_prefix=str(payload["knowledge_base_prefix"]),
            pricing_member=(
                str(payload["pricing_member"])
                if payload.get("pricing_member") is not None
                else None
            ),
            artifact_prefix=str(payload["artifact_prefix"]),
        )


def backup(config_path: Path, output_path: Path) -> Path:
    """Write a self-contained zip archive for the configuration at ``config_path``."""

    config = load_experiment_config(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = holdout_lock_path(config.dataset_path)
    has_lock = lock_path.is_file()
    pricing_path = config.pricing_path
    has_pricing = pricing_path is not None and pricing_path.is_file()

    manifest = Manifest(
        created_at=datetime.now(UTC).isoformat(),
        config_member="config.yaml",
        database_member="database/" + config.database_path.name,
        dataset_member="dataset/" + config.dataset_path.name,
        holdout_lock_member=("dataset/" + lock_path.name) if has_lock else None,
        knowledge_base_prefix="knowledge_base/",
        pricing_member=(
            "pricing/" + pricing_path.name if has_pricing and pricing_path is not None else None
        ),
        artifact_prefix="artifacts/",
    )

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(config_path, manifest.config_member)
        if config.database_path.is_file():
            archive.write(config.database_path, manifest.database_member)
        archive.write(config.dataset_path, manifest.dataset_member)
        if has_lock:
            assert manifest.holdout_lock_member is not None
            archive.write(lock_path, manifest.holdout_lock_member)
        if config.knowledge_base_path.is_dir():
            for file_path in sorted(config.knowledge_base_path.rglob("*")):
                if file_path.is_file():
                    relative = file_path.relative_to(config.knowledge_base_path)
                    archive.write(file_path, manifest.knowledge_base_prefix + relative.as_posix())
        if has_pricing:
            assert config.pricing_path is not None and manifest.pricing_member is not None
            archive.write(config.pricing_path, manifest.pricing_member)
        if config.artifact_dir.is_dir():
            for file_path in sorted(config.artifact_dir.rglob("*")):
                if file_path.is_file():
                    relative = file_path.relative_to(config.artifact_dir)
                    archive.write(file_path, manifest.artifact_prefix + relative.as_posix())
        archive.writestr(MANIFEST_NAME, json.dumps(manifest.to_json(), indent=2, sort_keys=True))
    return output_path


def restore(archive_path: Path, destination: Path) -> Path:
    """Extract ``archive_path`` into ``destination``; return the restored config path.

    The restored config file's ``database_path``, ``dataset_path``,
    ``knowledge_base_path``, ``artifact_dir``, and ``pricing_path`` (if any)
    are rewritten to point at the copies now under ``destination``, so it is
    immediately usable with ``rag-quality`` commands without touching the
    backup's original source paths.
    """

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = Manifest.from_json(json.loads(archive.read(MANIFEST_NAME)))
        archive.extractall(destination)

    config_path = destination / manifest.config_member
    database_path = destination / manifest.database_member
    dataset_path = destination / manifest.dataset_member
    knowledge_base_path = destination / manifest.knowledge_base_prefix
    artifact_dir = destination / manifest.artifact_prefix
    pricing_path = (
        destination / manifest.pricing_member if manifest.pricing_member is not None else None
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    payload["database_path"] = str(database_path)
    payload["dataset_path"] = str(dataset_path)
    payload["knowledge_base_path"] = str(knowledge_base_path)
    payload["artifact_dir"] = str(artifact_dir)
    if pricing_path is not None:
        payload["pricing_path"] = str(pricing_path)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    backup_parser = subcommands.add_parser(
        "backup", help="bundle a config's database, dataset, corpus, pricing, and artifacts"
    )
    backup_parser.add_argument("--config", required=True, type=Path)
    backup_parser.add_argument("--output", required=True, type=Path)

    restore_parser = subcommands.add_parser(
        "restore", help="extract a backup archive and rewrite its config's paths"
    )
    restore_parser.add_argument("--archive", required=True, type=Path)
    restore_parser.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "backup":
        path = backup(args.config, args.output)
        print(json.dumps({"archive": str(path.resolve())}))
    else:
        config_path = restore(args.archive, args.destination)
        print(json.dumps({"config": str(config_path.resolve())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
