"""Persistent job lifecycle: liveness, transitions, ledger journal, and schema."""

from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from rag_quality_lab.experiments.liveness import (
    LeaseHeartbeat,
    ProcessOwner,
    current_owner,
    probe_process,
)

from rag_quality_lab.domain.models import ExperimentIdentity, ExperimentStatus
from rag_quality_lab.experiments.store import SCHEMA_VERSION, ExperimentStore


def identity() -> ExperimentIdentity:
    return ExperimentIdentity(
        name="lifecycle",
        mode="live",
        commit_sha="abc123",
        dirty=False,
        dataset_hash="dataset-hash",
        prompt_hashes={"direct": "prompt-hash"},
        config={"seed": 42},
        random_seed=42,
        python_version="3.11.9",
    )


def dead_local_owner() -> ProcessOwner:
    """This PID with a different start time: the PID was reused, the owner is gone."""

    owner = current_owner()
    return ProcessOwner(pid=owner.pid, start_marker="not-this-process", host=owner.host)


def remote_owner() -> ProcessOwner:
    return ProcessOwner(pid=1, start_marker="remote", host=f"{socket.gethostname()}-other")


def test_probe_distinguishes_live_exited_and_reused_processes() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        marker = None
        for _ in range(100):
            owner = current_owner(child.pid)
            marker = owner.start_marker
            if marker is not None:
                break
            time.sleep(0.05)
        assert marker is not None
        assert probe_process(child.pid, marker) == "alive"
        assert probe_process(child.pid, "different-start") == "dead"
    finally:
        child.kill()
        child.wait()
    assert probe_process(child.pid, marker) == "dead"
    assert probe_process(os.getpid(), current_owner().start_marker) == "alive"


def test_orphaned_running_experiment_is_reaped_to_interrupted(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        orphan = store.create_experiment(identity(), owner=dead_local_owner())
        alive = store.create_experiment(identity())
        remote = store.create_experiment(identity(), owner=remote_owner())

        assert store.reap_orphans() == [orphan]
        assert store.reap_orphans(now=datetime.now(UTC) + timedelta(hours=1)) == [remote]

        assert store.get_experiment(orphan).status is ExperimentStatus.INTERRUPTED
        assert store.get_experiment(alive).status is ExperimentStatus.RUNNING
        assert store.get_experiment(remote).status is ExperimentStatus.INTERRUPTED


def test_heartbeat_renews_the_lease_of_a_remote_looking_owner(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    with ExperimentStore(database) as store:
        experiment_id = store.create_experiment(identity(), owner=remote_owner())
        token = store.lease_token(experiment_id)
        with LeaseHeartbeat(database, experiment_id, token, interval_seconds=0.05):
            time.sleep(0.3)
            later = datetime.now(UTC) + timedelta(seconds=30)
            assert store.reap_orphans(ttl_seconds=60, now=later) == []
        assert store.reap_orphans(ttl_seconds=60, now=later + timedelta(minutes=5)) == [
            experiment_id
        ]


def test_reaping_charges_dispatched_calls_and_releases_unsent_ones(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        experiment_id = store.create_experiment(identity(), owner=dead_local_owner())
        entry_ids = store.reserve_entries(
            experiment_id,
            task_key="case/arm/rag-001",
            config_id="arm",
            case_id="rag-001",
            reservations={"generation_with_repair": Decimal("0.3"), "judge": Decimal("0.2")},
        )
        with store.dispatch_journal() as journal:
            journal.mark_dispatched(entry_ids["generation_with_repair"])

        store.reap_orphans()

        entries = {entry.phase: entry for entry in store.ledger_entries(experiment_id)}
        assert entries["generation_with_repair"].state == "unknown"
        assert entries["generation_with_repair"].charged == Decimal("0.3")
        assert entries["judge"].state == "released"
        assert entries["judge"].charged == 0
        assert store.ledger_spent(experiment_id) == Decimal("0.3")
        record = store.get_experiment(experiment_id)
        assert [call.case_id for call in record.unknown_calls] == ["rag-001"]


def test_status_transitions_are_explicit(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        running = store.create_experiment(identity())
        assert store.request_cancel(running) is ExperimentStatus.RUNNING
        assert store.cancel_requested(running)
        store.finish_experiment(running, ExperimentStatus.CANCELLED)

        interrupted = store.create_experiment(identity(), owner=dead_local_owner())
        store.reap_orphans()
        assert store.request_cancel(interrupted) is ExperimentStatus.CANCELLED

        for terminal in (running, interrupted):
            with pytest.raises(ValueError, match="transition"):
                store.finish_experiment(terminal, ExperimentStatus.COMPLETED)
            with pytest.raises(ValueError, match="cancelled"):
                store.claim_for_resume(terminal)

        stuck = store.create_experiment(identity(), owner=dead_local_owner())
        store.request_cancel(stuck)
        assert store.get_experiment(stuck).status is ExperimentStatus.CANCELLED

        completed = store.create_experiment(identity())
        store.finish_experiment(completed, ExperimentStatus.COMPLETED)
        with pytest.raises(ValueError, match="completed"):
            store.request_cancel(completed)


def test_only_one_resumer_can_claim_an_interrupted_experiment(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "runs.sqlite3") as store:
        experiment_id = store.create_experiment(identity(), owner=dead_local_owner())
        store.reap_orphans()

        store.claim_for_resume(experiment_id)

        with pytest.raises(ValueError, match="running"):
            store.claim_for_resume(experiment_id)
        assert store.get_experiment(experiment_id).status is ExperimentStatus.RUNNING


LEGACY_SCHEMA = """
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    identity_json TEXT NOT NULL,
    summary_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE case_runs (
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
CREATE TABLE embedding_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    config_id TEXT NOT NULL,
    case_id TEXT,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def test_stage_one_database_is_migrated_and_stays_readable(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(LEGACY_SCHEMA)
    stale = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    for experiment_id, status in (("done", "completed"), ("crashed", "running")):
        connection.execute(
            "INSERT INTO experiments(id, status, identity_json, created_at) VALUES (?, ?, ?, ?)",
            (experiment_id, status, identity().model_dump_json(), stale),
        )
    connection.commit()
    connection.close()

    with ExperimentStore(database) as store:
        assert store.pragma("user_version") == SCHEMA_VERSION
        assert store.get_experiment("done").status is ExperimentStatus.COMPLETED
        assert store.reap_orphans() == ["crashed"]
        assert store.get_experiment("crashed").status is ExperimentStatus.INTERRUPTED
        assert store.ledger_spent("done") == 0


def test_newer_schema_is_rejected_with_an_explicit_error(tmp_path: Path) -> None:
    database = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(ValueError, match="newer"):
        ExperimentStore(database)
