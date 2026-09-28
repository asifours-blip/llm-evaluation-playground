"""A background run that finishes before start_run returns must stay completed."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from rag_quality_lab.web import service
from rag_quality_lab.web.app import create_app
from tests.web.conftest import write_mock_workbench_inputs
from tests.web.wsgi_client import call


def test_run_finishing_before_start_returns_is_not_reported_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_await = service._await_new_experiment_id

    def await_after_worker_finished(*args: Any, **kwargs: Any) -> str | None:
        # Force the fast-run ordering: the worker thread records its final
        # status before start_run gets to touch the state again.
        result = original_await(*args, **kwargs)
        for thread in threading.enumerate():
            if thread is not threading.current_thread() and thread.daemon:
                thread.join(timeout=30)
        return result

    monkeypatch.setattr(service, "_await_new_experiment_id", await_after_worker_finished)
    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    token = call(
        app, "POST", "/api/experiments/start", json_body={"config": str(config_path)}
    ).json()["token"]

    deadline = time.monotonic() + 5
    run_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        run_state = call(app, "GET", f"/api/runs/{token}").json()
        if run_state["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert run_state["status"] == "completed"
    assert run_state["experiment_id"]
