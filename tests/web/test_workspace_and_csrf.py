"""Workspace containment and CSRF defenses for the workbench server.

The server only binds loopback, but any page open in the operator's browser
-- including a malicious one -- can still reach it. These tests cover the
two defenses: every request-supplied path must resolve inside the server's
workspace, and every POST must be same-origin JSON.
"""

from __future__ import annotations

from pathlib import Path

from rag_quality_lab.web.app import create_app
from tests.web.conftest import write_mock_workbench_inputs
from tests.web.wsgi_client import call


def test_dataset_path_outside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    config_path = write_mock_workbench_inputs(outside)
    app = create_app(allow_live=False, workspace=workspace)

    response = call(
        app, "GET", "/api/dataset", query=f"dataset={outside / 'dataset.json'}"
    )
    assert response.status_code == 400

    response = call(app, "GET", "/api/config", query=f"config={config_path}")
    assert response.status_code == 400


def test_database_path_outside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_database = tmp_path / "outside" / "experiments.sqlite3"
    app = create_app(allow_live=False, workspace=workspace)

    response = call(app, "GET", "/api/experiments", query=f"database={outside_database}")
    assert response.status_code == 400


def test_relative_traversal_out_of_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    app = create_app(allow_live=False, workspace=workspace)

    response = call(app, "GET", "/api/dataset", query="dataset=../secret.json")
    assert response.status_code == 400


def test_start_body_config_outside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_config = write_mock_workbench_inputs(tmp_path / "outside")
    app = create_app(allow_live=False, workspace=workspace)

    response = call(
        app,
        "POST",
        "/api/experiments/start",
        json_body={"config": str(outside_config)},
    )
    assert response.status_code == 400


def test_path_inside_workspace_is_allowed(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(app, "GET", "/api/config", query=f"config={config_path}")
    assert response.status_code == 200


def test_post_with_non_json_content_type_is_refused(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    text_plain = call(
        app,
        "POST",
        "/api/experiments/start",
        raw_body=f'{{"config": "{config_path}"}}'.encode(),
        content_type="text/plain",
    )
    assert text_plain.status_code == 415

    form_encoded = call(
        app,
        "POST",
        "/api/experiments/start",
        raw_body=f"config={config_path}".encode(),
        content_type="application/x-www-form-urlencoded",
    )
    assert form_encoded.status_code == 415

    missing_content_type = call(
        app,
        "POST",
        "/api/experiments/start",
        raw_body=f'{{"config": "{config_path}"}}'.encode(),
        content_type=None,
    )
    assert missing_content_type.status_code == 415


def test_post_with_mismatched_origin_is_refused(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(
        app,
        "POST",
        "/api/experiments/start",
        json_body={"config": str(config_path)},
        origin="http://evil.example",
    )
    assert response.status_code == 403


def test_post_with_matching_origin_succeeds(tmp_path: Path) -> None:
    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    for origin in ("http://127.0.0.1:8765", "http://localhost:8765"):
        response = call(
            app,
            "POST",
            "/api/experiments/start",
            json_body={"config": str(config_path)},
            origin=origin,
        )
        assert response.status_code == 200, response.body


def test_post_without_origin_header_still_works(tmp_path: Path) -> None:
    """Non-browser clients (curl, the CLI's own tooling) send no Origin at
    all; the JSON content-type requirement alone still blocks cross-site
    form/script submissions, so a missing Origin header must not be refused."""

    config_path = write_mock_workbench_inputs(tmp_path)
    app = create_app(allow_live=False, workspace=tmp_path)

    response = call(
        app,
        "POST",
        "/api/experiments/start",
        json_body={"config": str(config_path)},
        origin=None,
    )
    assert response.status_code == 200
