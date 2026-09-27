"""SQLite repository for reproducible experiment records."""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import BaseModel

from rag_quality_lab.domain.models import (
    CaseResult,
    EmbeddingCallRecord,
    ExperimentIdentity,
    ExperimentRecord,
    ExperimentStatus,
    LedgerEntry,
)
from rag_quality_lab.experiments.liveness import (
    DEFAULT_LEASE_TTL_SECONDS,
    ProcessOwner,
    current_owner,
    probe_process,
)
from rag_quality_lab.metrics.calibration import AnnotationSnapshot, HumanAnnotation
from rag_quality_lab.metrics.judge import PairwiseComparisonRecord

# Version 1 is the unversioned stage-one schema (user_version 0); version 2
# adds execution leases, the budget ledger journal, and run attempts.
SCHEMA_VERSION = 2
PRAGMA_NAMES = {"journal_mode", "busy_timeout", "foreign_keys", "user_version"}
TERMINAL_STATUSES = {
    ExperimentStatus.COMPLETED,
    ExperimentStatus.FAILED,
    ExperimentStatus.BUDGET_EXCEEDED,
    ExperimentStatus.CANCELLED,
}
ALLOWED_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.RUNNING: frozenset(
        {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.BUDGET_EXCEEDED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.INTERRUPTED,
            ExperimentStatus.INCOMPLETE,
        }
    ),
    ExperimentStatus.INTERRUPTED: frozenset(
        {ExperimentStatus.RUNNING, ExperimentStatus.CANCELLED}
    ),
    # Re-entering RUNNING from INCOMPLETE requires retrying unknown calls.
    ExperimentStatus.INCOMPLETE: frozenset(
        {ExperimentStatus.RUNNING, ExperimentStatus.CANCELLED}
    ),
}
FINISH_STATUSES = TERMINAL_STATUSES | {ExperimentStatus.INCOMPLETE}
SettledState = Literal["settled", "released"]
Settlement = tuple[SettledState, Decimal]
EXPERIMENT_LEASE_COLUMNS = {
    "owner_pid": "INTEGER",
    "owner_start_marker": "TEXT",
    "owner_host": "TEXT",
    "lease_token": "TEXT",
    "heartbeat_at": "TEXT",
    "cancel_requested_at": "TEXT",
    "planned_task_count": "INTEGER",
}


class DispatchJournal:
    """Thread-safe gate consulted by worker threads before each request.

    It reads the cancellation flag and marks reservations as sent. It owns a
    separate connection so worker threads never share the store's
    connection; every mark is committed before the provider call starts.
    """

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self._lock = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def cancel_requested(self, experiment_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT cancel_requested_at FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
        return row is not None and row[0] is not None

    def mark_dispatched(self, entry_id: int) -> None:
        with self._lock, self.connection:
            cursor = self.connection.execute(
                """
                UPDATE ledger_entries SET state = 'dispatched', updated_at = ?
                WHERE id = ? AND state = 'reserved'
                """,
                (_utc_now(), entry_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"ledger entry {entry_id} is not reserved")


class ExperimentStore:
    """Single-writer experiment store with WAL-enabled readers."""

    def __init__(self, path: str | Path) -> None:
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = database_path
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA busy_timeout = 5000")
            self._create_schema()
        except BaseException:
            self.connection.close()
            raise

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

    def experiment_ids(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT id FROM experiments ORDER BY created_at, id"
        ).fetchall()
        return [str(row[0]) for row in rows]

    def create_experiment(
        self,
        identity: ExperimentIdentity,
        *,
        owner: ProcessOwner | None = None,
        planned_task_count: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Create a RUNNING experiment leased to ``owner`` (default: this process)."""

        experiment_id = str(uuid.uuid4())
        lease_owner = owner or current_owner()
        now = _utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO experiments(
                    id, status, identity_json, created_at, owner_pid,
                    owner_start_marker, owner_host, lease_token, heartbeat_at,
                    planned_task_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    ExperimentStatus.RUNNING.value,
                    _canonical_json(identity),
                    now,
                    lease_owner.pid,
                    lease_owner.start_marker,
                    lease_owner.host,
                    str(uuid.uuid4()),
                    now,
                    planned_task_count,
                ),
            )
            self._start_attempt(experiment_id, 1, "run", lease_owner, metadata or {})
        return experiment_id

    def lease_token(self, experiment_id: str) -> str:
        return str(self._row(experiment_id)["lease_token"] or "")

    def current_attempt(self, experiment_id: str) -> int:
        row = self.connection.execute(
            "SELECT MAX(attempt) FROM experiment_runs WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return int(row[0] or 1)

    def reap_orphans(
        self,
        *,
        ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
        now: datetime | None = None,
    ) -> list[str]:
        """Mark RUNNING experiments whose owner is gone as INTERRUPTED.

        A local owner is gone when its PID no longer runs the process with the
        recorded start time. Owners on other hosts, legacy rows without an
        owner, and platforms without a start-time probe fall back to the
        heartbeat lease expiring. Outstanding reservations are resolved in the
        same transaction: unsent ones are released, sent ones become unknown
        and are charged at their full reservation. An experiment whose
        cancellation was requested becomes CANCELLED instead.
        """

        moment = now or datetime.now(UTC)
        rows = self.connection.execute(
            """
            SELECT id, created_at, owner_pid, owner_start_marker, owner_host,
                   lease_token, heartbeat_at
            FROM experiments WHERE status = 'running' ORDER BY created_at, id
            """
        ).fetchall()
        reaped: list[str] = []
        for row in rows:
            if _owner_is_gone(row, moment, ttl_seconds) and self._interrupt(
                str(row["id"]), row["lease_token"]
            ):
                reaped.append(str(row["id"]))
        return reaped

    def mark_interrupted(self, experiment_id: str) -> None:
        """Interrupt an experiment owned by this process, e.g. after Ctrl+C."""

        self._interrupt(experiment_id, self._row(experiment_id)["lease_token"])

    def request_cancel(self, experiment_id: str) -> ExperimentStatus:
        """Cancel an experiment; a live owner stops before claiming new tasks."""

        self.reap_orphans()
        status = self._status(experiment_id)
        if status is ExperimentStatus.CANCELLED:
            return status
        if status is ExperimentStatus.RUNNING:
            with self.connection:
                self.connection.execute(
                    """
                    UPDATE experiments
                    SET cancel_requested_at = COALESCE(cancel_requested_at, ?)
                    WHERE id = ? AND status = 'running'
                    """,
                    (_utc_now(), experiment_id),
                )
            return self._status(experiment_id)
        if status in {ExperimentStatus.INTERRUPTED, ExperimentStatus.INCOMPLETE}:
            self._transition(experiment_id, status, ExperimentStatus.CANCELLED)
            return self._status(experiment_id)
        raise ValueError(f"cannot cancel experiment {experiment_id}: status is {status}")

    def cancel_requested(self, experiment_id: str) -> bool:
        return self._row(experiment_id)["cancel_requested_at"] is not None

    def claim_for_resume(
        self,
        experiment_id: str,
        *,
        owner: ProcessOwner | None = None,
        metadata: dict[str, object] | None = None,
        retry_unknown: bool = False,
    ) -> int:
        """Atomically lease a resumable experiment to a resumer; return its attempt.

        INTERRUPTED experiments are always resumable; INCOMPLETE ones only
        when ``retry_unknown`` re-sends their unknown-outcome case arms.
        """

        lease_owner = owner or current_owner()
        # Both placeholders name INTERRUPTED unless INCOMPLETE is also resumable.
        resumable = ("interrupted", "incomplete" if retry_unknown else "interrupted")
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE experiments
                SET status = 'running', owner_pid = ?, owner_start_marker = ?,
                    owner_host = ?, lease_token = ?, heartbeat_at = ?, finished_at = NULL
                WHERE id = ? AND status IN (?, ?)
                """,
                (
                    lease_owner.pid,
                    lease_owner.start_marker,
                    lease_owner.host,
                    str(uuid.uuid4()),
                    _utc_now(),
                    experiment_id,
                    *resumable,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    resume_refusal(experiment_id, self._status(experiment_id))
                )
            attempt = self.current_attempt(experiment_id) + 1
            self._start_attempt(
                experiment_id, attempt, "resume", lease_owner, metadata or {}
            )
        return attempt

    def reserve_entries(
        self,
        experiment_id: str,
        *,
        task_key: str,
        config_id: str,
        case_id: str | None,
        reservations: Mapping[str, Decimal],
    ) -> dict[str, int]:
        """Journal the budget reserved for each phase of one task before it runs."""

        self._require_running(experiment_id)
        attempt = self.current_attempt(experiment_id)
        now = _utc_now()
        entry_ids: dict[str, int] = {}
        with self.connection:
            for phase, amount in reservations.items():
                cursor = self.connection.execute(
                    """
                    INSERT INTO ledger_entries(
                        experiment_id, task_key, config_id, case_id, phase, attempt,
                        state, reserved_amount, charged_amount, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'reserved', ?, '0', ?, ?)
                    """,
                    (
                        experiment_id,
                        task_key,
                        config_id,
                        case_id,
                        phase,
                        attempt,
                        str(amount),
                        now,
                        now,
                    ),
                )
                if cursor.lastrowid is None:
                    raise ValueError("ledger entry was not inserted")
                entry_ids[phase] = cursor.lastrowid
        return entry_ids

    def dispatch_journal(self) -> DispatchJournal:
        return DispatchJournal(self.path)

    def settle_entries(
        self, experiment_id: str, settlements: Mapping[int, Settlement]
    ) -> None:
        self._require_running(experiment_id)
        with self.connection:
            self._settle(settlements)

    def commit_case_outcome(
        self,
        experiment_id: str,
        result: CaseResult,
        embedding_calls: Sequence[EmbeddingCallRecord] = (),
        settlements: Mapping[int, Settlement] | None = None,
    ) -> None:
        """Atomically persist one case outcome, its embeddings, and its settlement.

        Either the case becomes done and its reservations settle together, or
        nothing changes and a later resume treats sent reservations as unknown.
        """

        self._require_running(experiment_id)
        try:
            with self.connection:
                for record in embedding_calls:
                    self._insert_embedding_call(experiment_id, record)
                self._settle(settlements or {})
                self._insert_case_result(experiment_id, result)
        except sqlite3.IntegrityError as error:
            if "UNIQUE" in str(error).upper():
                raise ValueError("duplicate case result") from error
            raise

    def commit_embedding_outcome(
        self,
        experiment_id: str,
        record: EmbeddingCallRecord,
        settlements: Mapping[int, Settlement] | None = None,
    ) -> None:
        """Atomically persist one index embedding request and its settlement."""

        self._require_running(experiment_id)
        with self.connection:
            self._insert_embedding_call(experiment_id, record)
            self._settle(settlements or {})

    def record_case_result(self, experiment_id: str, result: CaseResult) -> None:
        self.commit_case_outcome(experiment_id, result)

    def record_embedding_call(
        self, experiment_id: str, record: EmbeddingCallRecord
    ) -> None:
        """Persist one budgeted live embedding request batch."""

        self.commit_embedding_outcome(experiment_id, record)

    def ledger_entries(self, experiment_id: str) -> list[LedgerEntry]:
        rows = self.connection.execute(
            """
            SELECT task_key, config_id, case_id, phase, attempt, state,
                   reserved_amount, charged_amount
            FROM ledger_entries WHERE experiment_id = ? ORDER BY id
            """,
            (experiment_id,),
        ).fetchall()
        return [
            LedgerEntry(
                task_key=row["task_key"],
                config_id=row["config_id"],
                case_id=row["case_id"],
                phase=row["phase"],
                attempt=row["attempt"],
                state=row["state"],
                reserved=Decimal(row["reserved_amount"]),
                charged=Decimal(row["charged_amount"]),
            )
            for row in rows
        ]

    def ledger_spent(self, experiment_id: str) -> Decimal:
        """Return settled plus unknown charges; open reservations are not spend."""

        self._status(experiment_id)
        return sum(
            (entry.charged for entry in self.ledger_entries(experiment_id)),
            start=Decimal("0"),
        )

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

    def recorded_case_keys(self, experiment_id: str) -> set[tuple[str, str]]:
        """Return (case_id, config_id) pairs with a persisted outcome of any status."""

        rows = self.connection.execute(
            "SELECT case_id, config_id FROM case_runs WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        return {(str(row[0]), str(row[1])) for row in rows}

    def unknown_task_keys(self, experiment_id: str) -> set[str]:
        return {
            entry.task_key
            for entry in self.ledger_entries(experiment_id)
            if entry.state == "unknown"
        }

    def progress(self, experiment_id: str) -> dict[str, Any]:
        """Summarize lifecycle, case-arm progress, and spend for status views."""

        row = self._row(experiment_id)
        record = self.get_experiment(experiment_id)
        done = self.recorded_case_keys(experiment_id)
        unknown = {
            (call.case_id, call.config_id)
            for call in record.unknown_calls
            if call.case_id is not None
        } - done
        if self.ledger_entries(experiment_id):
            spent = self.ledger_spent(experiment_id)
        else:
            # Mock runs and pre-ledger databases: report the recorded costs.
            spent = sum(
                (result.cost for result in record.case_results), start=Decimal("0")
            ) + sum((call.cost for call in record.embedding_calls), start=Decimal("0"))
        total = row["planned_task_count"]
        return {
            "id": experiment_id,
            "name": record.identity.name,
            "mode": record.identity.mode,
            "status": record.status.value,
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
            "progress": {
                "done": len(done),
                "total": int(total) if total is not None else len(done),
                "unknown": len(unknown),
            },
            "spent": str(spent),
            "unknown_cost": str(
                sum((call.charged for call in record.unknown_calls), start=Decimal("0"))
            ),
            "cancel_requested": row["cancel_requested_at"] is not None,
            "owner": {
                "pid": row["owner_pid"],
                "host": row["owner_host"],
                "heartbeat_at": row["heartbeat_at"],
            },
            "attempts": self.current_attempt(experiment_id),
        }

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
        if status not in FINISH_STATUSES:
            raise ValueError("experiment can only finish in a terminal or incomplete status")
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
            self._end_attempt(experiment_id, status)

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
        embedding_rows = self.connection.execute(
            """
            SELECT payload_json FROM embedding_calls
            WHERE experiment_id = ?
            ORDER BY id
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
            embedding_calls=[
                EmbeddingCallRecord.model_validate_json(embedding_row["payload_json"])
                for embedding_row in embedding_rows
            ],
            unknown_calls=[
                entry
                for entry in self.ledger_entries(experiment_id)
                if entry.state == "unknown"
            ],
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

    def _row(self, experiment_id: str) -> sqlite3.Row:
        row: sqlite3.Row | None = self.connection.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return row

    def _transition(
        self,
        experiment_id: str,
        current: ExperimentStatus,
        target: ExperimentStatus,
    ) -> bool:
        """Move one experiment between legal states; False if it changed meanwhile."""

        if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
            raise ValueError(f"illegal experiment status transition: {current} -> {target}")
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE experiments SET status = ?, finished_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    target.value,
                    _utc_now() if target in TERMINAL_STATUSES else None,
                    experiment_id,
                    current.value,
                ),
            )
            changed = cursor.rowcount == 1
            if changed:
                self._end_attempt(experiment_id, target)
        return changed

    def _interrupt(self, experiment_id: str, lease_token: object) -> bool:
        """Resolve open reservations and leave RUNNING, guarded by the lease token."""

        row = self._row(experiment_id)
        target = (
            ExperimentStatus.CANCELLED
            if row["cancel_requested_at"] is not None
            else ExperimentStatus.INTERRUPTED
        )
        now = _utc_now()
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE experiments SET status = ?, finished_at = ?
                WHERE id = ? AND status = 'running' AND COALESCE(lease_token, '') = ?
                """,
                (
                    target.value,
                    now if target is ExperimentStatus.CANCELLED else None,
                    experiment_id,
                    str(lease_token or ""),
                ),
            )
            if cursor.rowcount != 1:
                return False
            self.connection.execute(
                """
                UPDATE ledger_entries SET state = 'released', updated_at = ?
                WHERE experiment_id = ? AND state = 'reserved'
                """,
                (now, experiment_id),
            )
            self.connection.execute(
                """
                UPDATE ledger_entries
                SET state = 'unknown', charged_amount = reserved_amount, updated_at = ?
                WHERE experiment_id = ? AND state = 'dispatched'
                """,
                (now, experiment_id),
            )
            self._end_attempt(experiment_id, target)
        return True

    def _start_attempt(
        self,
        experiment_id: str,
        attempt: int,
        kind: str,
        owner: ProcessOwner,
        metadata: dict[str, object],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO experiment_runs(
                experiment_id, attempt, kind, owner_pid, owner_start_marker,
                owner_host, started_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                attempt,
                kind,
                owner.pid,
                owner.start_marker,
                owner.host,
                _utc_now(),
                _canonical_payload(metadata),
            ),
        )

    def _end_attempt(self, experiment_id: str, status: ExperimentStatus) -> None:
        self.connection.execute(
            """
            UPDATE experiment_runs SET ended_at = ?, end_status = ?
            WHERE experiment_id = ? AND ended_at IS NULL
            """,
            (_utc_now(), status.value, experiment_id),
        )

    def _settle(self, settlements: Mapping[int, Settlement]) -> None:
        now = _utc_now()
        for entry_id, (state, charged) in settlements.items():
            cursor = self.connection.execute(
                """
                UPDATE ledger_entries SET state = ?, charged_amount = ?, updated_at = ?
                WHERE id = ? AND state IN ('reserved', 'dispatched')
                """,
                (state, str(charged), now, entry_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"ledger entry {entry_id} is not outstanding")

    def _insert_case_result(self, experiment_id: str, result: CaseResult) -> None:
        case_run_id = str(uuid.uuid4())
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

    def _insert_embedding_call(
        self, experiment_id: str, record: EmbeddingCallRecord
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO embedding_calls(
                experiment_id, phase, config_id, case_id, status,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                record.phase,
                record.config_id,
                record.case_id,
                record.status,
                _canonical_json(record),
                _utc_now(),
            ),
        )

    def _create_schema(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"database schema version {version} at {self.path} is newer than "
                f"supported version {SCHEMA_VERSION}; upgrade rag-quality-lab to open it"
            )
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

            CREATE TABLE IF NOT EXISTS embedding_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                phase TEXT NOT NULL,
                config_id TEXT NOT NULL,
                case_id TEXT,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pairwise_comparisons (
                id TEXT PRIMARY KEY,
                baseline_experiment_id TEXT NOT NULL REFERENCES experiments(id),
                candidate_experiment_id TEXT NOT NULL REFERENCES experiments(id),
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ledger_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                task_key TEXT NOT NULL,
                config_id TEXT NOT NULL,
                case_id TEXT,
                phase TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(
                    state IN ('reserved', 'dispatched', 'settled', 'released', 'unknown')
                ),
                reserved_amount TEXT NOT NULL,
                charged_amount TEXT NOT NULL DEFAULT '0',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS ledger_entries_by_experiment
            ON ledger_entries(experiment_id, state);

            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                attempt INTEGER NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('run', 'resume')),
                owner_pid INTEGER,
                owner_start_marker TEXT,
                owner_host TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                end_status TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(experiment_id, attempt)
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
        for column, column_type in EXPERIMENT_LEASE_COLUMNS.items():
            if column not in experiment_columns:
                self.connection.execute(
                    f"ALTER TABLE experiments ADD COLUMN {column} {column_type}"
                )
        if version < SCHEMA_VERSION:
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.connection.commit()


def resume_refusal(experiment_id: str, status: ExperimentStatus) -> str:
    """Explain why an experiment in ``status`` cannot be resumed as requested."""

    prefix = f"cannot resume experiment {experiment_id}: status is {status}"
    if status is ExperimentStatus.INCOMPLETE:
        return (
            f"{prefix}; its remaining case arms have unknown outcomes. Pass "
            "--retry-unknown to re-send them at additional cost, or cancel it"
        )
    if status is ExperimentStatus.RUNNING:
        return (
            f"{prefix} and its owning process is still alive; cancel it or wait "
            "for it to stop"
        )
    return (
        f"{prefix}; only interrupted experiments, or incomplete ones with "
        "--retry-unknown, can be resumed"
    )


def _owner_is_gone(row: sqlite3.Row, now: datetime, ttl_seconds: float) -> bool:
    pid = row["owner_pid"]
    if pid is not None and row["owner_host"] == socket.gethostname():
        state = probe_process(int(pid), row["owner_start_marker"])
        if state != "unknown":
            return state == "dead"
    last_seen = datetime.fromisoformat(str(row["heartbeat_at"] or row["created_at"]))
    return (now - last_seen).total_seconds() > ttl_seconds


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
