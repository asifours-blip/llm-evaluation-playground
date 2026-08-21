"""SQLite repository for reproducible experiment records."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from rag_quality_lab.domain.models import (
    CaseResult,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
)
from rag_quality_lab.metrics.calibration import AnnotationSnapshot, HumanAnnotation
from rag_quality_lab.metrics.judge import PairwiseComparisonRecord

PRAGMA_NAMES = {"journal_mode", "busy_timeout", "foreign_keys"}
TERMINAL_STATUSES = {
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.BUDGET_EXCEEDED,
}


class ExperimentStore:
    """Single-writer experiment store with WAL-enabled readers."""

    def __init__(self, path: str | Path) -> None:
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._create_schema()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def pragma(self, name: str) -> object:
        if name not in PRAGMA_NAMES:
            raise ValueError(f"unsupported pragma: {name}")
        row = self.connection.execute(f"PRAGMA {name}").fetchone()
        if row is None:
            raise ValueError(f"pragma returned no value: {name}")
        return row[0]

    def table_names(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def resolve_experiment_id(self, identifier: str) -> str:
        """Resolve an explicit ID or the newest live experiment alias."""

        if identifier != "latest-live":
            self._status(identifier)
            return identifier
        rows = self.connection.execute(
            "SELECT id, identity_json FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        for row in rows:
            identity = ExperimentIdentity.model_validate_json(row["identity_json"])
            if identity.mode == "live":
                return str(row["id"])
        raise KeyError("no live experiment exists")

    def create_experiment(self, identity: ExperimentIdentity) -> str:
        experiment_id = str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO experiments(id, status, identity_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    ExperimentStatus.RUNNING.value,
                    _canonical_json(identity),
                    _utc_now(),
                ),
            )
        return experiment_id

    def record_case_result(self, experiment_id: str, result: CaseResult) -> None:
        self._require_running(experiment_id)
        case_run_id = str(uuid.uuid4())
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO case_runs(
                        id, experiment_id, case_id, config_id, model, status,
                        payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_run_id,
                        experiment_id,
                        result.case_id,
                        result.config_id,
                        result.model,
                        result.status,
                        _canonical_json(result),
                        _utc_now(),
                    ),
                )
                for rank, hit in enumerate(result.retrieval_hits, start=1):
                    self.connection.execute(
                        """
                        INSERT INTO retrieval_hits(
                            case_run_id, rank, document_id, chunk_id, score, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            case_run_id,
                            rank,
                            hit.chunk.document_id,
                            hit.chunk.id,
                            hit.score,
                            _canonical_json(hit),
                        ),
                    )
                for metric_name, metric_value in sorted(result.metrics.items()):
                    self.connection.execute(
                        """
                        INSERT INTO metric_results(case_run_id, name, value)
                        VALUES (?, ?, ?)
                        """,
                        (case_run_id, metric_name, metric_value),
                    )
        except sqlite3.IntegrityError as error:
            if "UNIQUE" in str(error).upper():
                raise ValueError("duplicate case result") from error
            raise

    def completed_case_keys(self, experiment_id: str) -> set[tuple[str, str, str]]:
        rows = self.connection.execute(
            """
            SELECT case_id, config_id, model
            FROM case_runs
            WHERE experiment_id = ? AND status = 'completed'
            """,
            (experiment_id,),
        ).fetchall()
        return {(str(row[0]), str(row[1]), str(row[2])) for row in rows}

    def record_artifact(
        self,
        experiment_id: str,
        *,
        kind: str,
        path: str,
        sha256: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._status(experiment_id)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO artifacts(experiment_id, kind, path, sha256, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    kind,
                    path,
                    sha256,
                    _canonical_payload(metadata or {}),
                ),
            )

    def record_human_annotations(
        self, experiment_id: str, annotations: list[HumanAnnotation]
    ) -> None:
        self._status(experiment_id)
        try:
            with self.connection:
                for annotation in annotations:
                    self.connection.execute(
                        """
                        INSERT INTO human_annotations(
                            experiment_id, case_id, human_score, payload_json
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            experiment_id,
                            annotation.sample_id,
                            annotation.human_score,
                            _canonical_json(annotation),
                        ),
                    )
        except sqlite3.IntegrityError as error:
            if "UNIQUE" in str(error).upper():
                raise ValueError("duplicate human annotation") from error
            raise

    def get_human_annotations(self, experiment_id: str) -> list[HumanAnnotation]:
        rows = self.connection.execute(
            """
            SELECT payload_json FROM human_annotations
            WHERE experiment_id = ? ORDER BY case_id
            """,
            (experiment_id,),
        ).fetchall()
        return [HumanAnnotation.model_validate_json(row[0]) for row in rows]

    def record_annotation_snapshots(
        self, experiment_id: str, snapshots: list[AnnotationSnapshot]
    ) -> None:
        """Persist the private identity/hash mapping for one blind export."""

        self._status(experiment_id)
        with self.connection:
            for snapshot in snapshots:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO annotation_snapshots(
                        experiment_id, sample_id, payload_json
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        experiment_id,
                        snapshot.sample_id,
                        _canonical_json(snapshot),
                    ),
                )

    def get_annotation_snapshots(
        self, experiment_id: str
    ) -> list[AnnotationSnapshot]:
        rows = self.connection.execute(
            """
            SELECT payload_json FROM annotation_snapshots
            WHERE experiment_id = ? ORDER BY sample_id
            """,
            (experiment_id,),
        ).fetchall()
        return [AnnotationSnapshot.model_validate_json(row[0]) for row in rows]

    def record_pairwise_comparison(self, record: PairwiseComparisonRecord) -> None:
        """Persist one complete two-order judge comparison."""

        self._status(record.baseline_experiment_id)
        self._status(record.candidate_experiment_id)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO pairwise_comparisons(
                    id, baseline_experiment_id, candidate_experiment_id, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.baseline_experiment_id,
                    record.candidate_experiment_id,
                    _canonical_json(record),
                ),
            )

    def get_pairwise_comparison(self, comparison_id: str) -> PairwiseComparisonRecord:
        row = self.connection.execute(
            "SELECT payload_json FROM pairwise_comparisons WHERE id = ?",
            (comparison_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown pairwise comparison: {comparison_id}")
        return PairwiseComparisonRecord.model_validate_json(row[0])

    def finish_experiment(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        *,
        summary: dict[str, float] | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("experiment can only finish in a terminal status")
        current = self._status(experiment_id)
        if current is not ExperimentStatus.RUNNING:
            raise ValueError(f"illegal experiment status transition: {current} -> {status}")
        with self.connection:
            self.connection.execute(
                """
                UPDATE experiments
                SET status = ?, finished_at = ?, summary_json = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    _utc_now(),
                    _canonical_payload(summary or {}),
                    experiment_id,
                ),
            )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        row = self.connection.execute(
            "SELECT status, identity_json, summary_json FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        result_rows = self.connection.execute(
            """
            SELECT payload_json FROM case_runs
            WHERE experiment_id = ?
            ORDER BY created_at, case_id, config_id, model
            """,
            (experiment_id,),
        ).fetchall()
        return ExperimentRecord(
            id=experiment_id,
            identity=ExperimentIdentity.model_validate_json(row["identity_json"]),
            status=ExperimentStatus(row["status"]),
            case_results=[
                CaseResult.model_validate_json(result_row["payload_json"])
                for result_row in result_rows
            ],
            summary=json.loads(row["summary_json"]),
        )

    def _status(self, experiment_id: str) -> ExperimentStatus:
        row = self.connection.execute(
            "SELECT status FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return ExperimentStatus(row[0])

    def _require_running(self, experiment_id: str) -> None:
        status = self._status(experiment_id)
        if status is not ExperimentStatus.RUNNING:
            raise ValueError(f"experiment is not running: {status}")

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                identity_json TEXT NOT NULL,
                summary_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS case_runs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                case_id TEXT NOT NULL,
                config_id TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(experiment_id, case_id, config_id, model)
            );

            CREATE TABLE IF NOT EXISTS retrieval_hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_run_id TEXT NOT NULL REFERENCES case_runs(id) ON DELETE CASCADE,
                rank INTEGER NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                score REAL NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(case_run_id, rank)
            );

            CREATE TABLE IF NOT EXISTS metric_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_run_id TEXT NOT NULL REFERENCES case_runs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                UNIQUE(case_run_id, name)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS human_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                case_id TEXT NOT NULL,
                human_score INTEGER NOT NULL CHECK(human_score BETWEEN 1 AND 5),
                payload_json TEXT NOT NULL,
                UNIQUE(experiment_id, case_id)
            );

            CREATE TABLE IF NOT EXISTS annotation_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                sample_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(experiment_id, sample_id)
            );

            CREATE TABLE IF NOT EXISTS pairwise_comparisons (
                id TEXT PRIMARY KEY,
                baseline_experiment_id TEXT NOT NULL REFERENCES experiments(id),
                candidate_experiment_id TEXT NOT NULL REFERENCES experiments(id),
                payload_json TEXT NOT NULL
            );
            """
        )
        experiment_columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "summary_json" not in experiment_columns:
            self.connection.execute(
                "ALTER TABLE experiments ADD COLUMN summary_json TEXT NOT NULL DEFAULT '{}'"
            )


def _canonical_json(model: BaseModel) -> str:
    return _canonical_payload(model.model_dump(mode="json"))


def _canonical_payload(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
