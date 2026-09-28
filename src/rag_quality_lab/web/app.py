"""A stdlib-only WSGI workbench for browsing datasets, running experiments, and
downloading reports.

The app is a thin JSON API plus one static page (``static/index.html``); every
handler below delegates to :mod:`rag_quality_lab.web.service`, which in turn
calls the existing config loaders, :mod:`rag_quality_lab.experiments`, and
:mod:`rag_quality_lab.reporting` modules directly. No business logic is
duplicated here.

Live experiment runs are refused unless the server was started with
``allow_live=True`` *and* the request explicitly confirms it, mirroring the
CLI's ``--confirm-live-run`` gate.

Two defenses keep a page open in the operator's own browser from using this
server as a confused deputy (the server only binds loopback, but any local
page -- including a malicious one -- can still reach it):

- Every request-supplied path (dataset, config, database, output, ...) is
  resolved and must stay inside the server's ``workspace`` directory
  (``--workspace``, default the current working directory); anything else is
  refused with 400 before it is ever opened.
- Every POST (the only requests that start runs or write files) must carry
  ``Content-Type: application/json`` -- which a cross-origin form or ``img``/
  ``script`` tag cannot send without triggering a CORS preflight this server
  does not answer -- and, if the browser also sends an ``Origin`` header, it
  must name this server's own origin.
"""

from __future__ import annotations

import json
import mimetypes
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
from urllib.parse import parse_qs
from wsgiref.simple_server import WSGIServer, make_server

from rag_quality_lab.web import service

if TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse as WSGIStartResponse

STATIC_DIR = Path(__file__).parent / "static"

WSGIEnviron = dict[str, Any]
StartResponse: TypeAlias = "WSGIStartResponse"
Params = dict[str, str]
Query = dict[str, list[str]]
Handler = Callable[[WSGIEnviron, Params, Query], "Response"]


class PathEscapesWorkspace(ValueError):
    """Raised when a request-supplied path would resolve outside the workspace."""


class UnsupportedContentType(ValueError):
    """Raised when a POST body's Content-Type is not application/json."""


class OriginNotAllowed(PermissionError):
    """Raised when a POST request's Origin header names a different server."""


class Response:
    """A small, explicit WSGI response: status, headers, and a body."""

    def __init__(
        self,
        body: bytes,
        *,
        status: str = "200 OK",
        content_type: str = "application/json",
    ) -> None:
        self.body = body
        self.status = status
        self.headers = [("Content-Type", content_type), ("Content-Length", str(len(body)))]

    @classmethod
    def json(cls, payload: object, *, status: str = "200 OK") -> Response:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return cls(text.encode("utf-8"), status=status, content_type="application/json")

    @classmethod
    def error(cls, status: str, message: str) -> Response:
        return cls.json({"error": message}, status=status)

    @classmethod
    def file(cls, path: Path) -> Response:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return cls(path.read_bytes(), content_type=content_type)


_ROUTES: list[tuple[str, re.Pattern[str], str]] = []


def route(method: str, pattern: str, name: str) -> None:
    _ROUTES.append((method, re.compile(f"^{pattern}$"), name))


# Every GET route below must be read-only: no writing files, no starting or
# resuming a run, no cancelling one. Anything with a side effect is POST and
# goes through _check_post_safety() (same-origin, application/json only) in
# _dispatch(); a GET on a POST-only path gets 405, not a quiet fallback.
route("GET", r"/", "index")
route("GET", r"/api/dataset", "dataset")
route("GET", r"/api/config", "config")
route("GET", r"/api/precheck", "precheck")
route("GET", r"/api/experiments", "list_experiments")
route("GET", r"/api/experiments/(?P<experiment_id>[^/]+)", "experiment_detail")
route("GET", r"/api/experiments/(?P<experiment_id>[^/]+)/results", "case_results")
route("POST", r"/api/experiments/start", "start")
route("POST", r"/api/experiments/(?P<experiment_id>[^/]+)/resume", "resume")
route("POST", r"/api/experiments/(?P<experiment_id>[^/]+)/cancel", "cancel")
route("POST", r"/api/experiments/(?P<experiment_id>[^/]+)/report", "report")
route("GET", r"/api/runs/(?P<token>[^/]+)", "run_status")
route("POST", r"/api/compare", "compare")
route("GET", r"/api/artifacts/download", "download")


WSGIApp = Callable[[WSGIEnviron, StartResponse], Iterable[bytes]]


def create_app(*, allow_live: bool = False, workspace: Path | str | None = None) -> WSGIApp:
    """Build the WSGI application.

    ``allow_live=False`` (the default) refuses every live run regardless of
    what a request asks for. ``workspace`` (default: the current working
    directory) is the root every request-supplied path must resolve inside;
    a request naming a path outside it is refused with 400 before it is
    opened.
    """

    resolved_workspace = Path(workspace).resolve() if workspace is not None else Path.cwd()
    manager = service.RunManager(allow_live=allow_live)
    handlers: dict[str, Handler] = {
        "index": lambda environ, params, query: _handle_index(),
        "dataset": lambda environ, params, query: _handle_dataset(query, resolved_workspace),
        "config": lambda environ, params, query: _handle_config(query, resolved_workspace),
        "precheck": lambda environ, params, query: _handle_precheck(query, resolved_workspace),
        "list_experiments": (
            lambda environ, params, query: _handle_list_experiments(query, resolved_workspace)
        ),
        "experiment_detail": (
            lambda environ, params, query: _handle_experiment_detail(
                params, query, resolved_workspace
            )
        ),
        "case_results": (
            lambda environ, params, query: _handle_case_results(
                params, query, resolved_workspace
            )
        ),
        "report": (
            lambda environ, params, query: _handle_report(environ, params, resolved_workspace)
        ),
        "compare": lambda environ, params, query: _handle_compare(environ, resolved_workspace),
        "download": lambda environ, params, query: _handle_download(query, resolved_workspace),
        "start": (
            lambda environ, params, query: _handle_start(environ, manager, resolved_workspace)
        ),
        "resume": (
            lambda environ, params, query: _handle_resume(
                environ, params, manager, resolved_workspace
            )
        ),
        "cancel": (
            lambda environ, params, query: _handle_cancel(environ, manager, resolved_workspace)
        ),
        "run_status": lambda environ, params, query: _handle_run_status(params, manager),
    }

    def app(environ: WSGIEnviron, start_response: StartResponse) -> Iterable[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        query = parse_qs(environ.get("QUERY_STRING", ""))
        response = _dispatch(method, path, environ, query, handlers)
        start_response(response.status, response.headers)
        return [response.body]

    return app


def _dispatch(
    method: str,
    path: str,
    environ: WSGIEnviron,
    query: Query,
    handlers: dict[str, Handler],
) -> Response:
    methods_for_path: set[str] = set()
    for route_method, pattern, name in _ROUTES:
        match = pattern.match(path)
        if match is None:
            continue
        methods_for_path.add(route_method)
        if route_method != method:
            continue
        try:
            if method == "POST":
                _check_post_safety(environ)
            return handlers[name](environ, match.groupdict(), query)
        except UnsupportedContentType as error:
            return Response.error("415 Unsupported Media Type", str(error))
        except OriginNotAllowed as error:
            return Response.error("403 Forbidden", str(error))
        except service.LiveRunNotAllowed as error:
            return Response.error("403 Forbidden", str(error))
        except PathEscapesWorkspace as error:
            return Response.error("400 Bad Request", str(error))
        except service.PathEscapesArtifactDir as error:
            return Response.error("400 Bad Request", str(error))
        except KeyError as error:
            return Response.error("404 Not Found", str(error))
        except FileNotFoundError as error:
            return Response.error("404 Not Found", str(error))
        except (ValueError, PermissionError) as error:
            return Response.error("400 Bad Request", str(error))
    if methods_for_path:
        return Response.error(
            "405 Method Not Allowed",
            f"{method} not allowed for {path}; use {', '.join(sorted(methods_for_path))}",
        )
    return Response.error("404 Not Found", f"no route for {method} {path}")


def _check_post_safety(environ: WSGIEnviron) -> None:
    """Reject cross-site POSTs: browsers cannot send JSON-typed, same-origin
    POSTs without cooperation from this server (no CORS headers are ever
    sent), so requiring both is enough to refuse a page on another origin."""

    content_type = environ.get("CONTENT_TYPE", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise UnsupportedContentType(
            "POST requests require Content-Type: application/json, got: "
            + (content_type or "(none)")
        )
    origin = environ.get("HTTP_ORIGIN")
    if origin is not None and origin not in _allowed_origins(environ):
        raise OriginNotAllowed(f"Origin not allowed: {origin}")


def _allowed_origins(environ: WSGIEnviron) -> set[str]:
    host_header = environ.get("HTTP_HOST", "")
    if ":" in host_header:
        port = host_header.rsplit(":", 1)[1]
    else:
        port = str(environ.get("SERVER_PORT") or "80")
    return {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


def _handle_index() -> Response:
    return Response.file(STATIC_DIR / "index.html")


def _handle_dataset(query: Query, workspace: Path) -> Response:
    dataset_path = _required_workspace_path(query, "dataset", workspace)
    return Response.json(service.dataset_view(dataset_path))


def _handle_config(query: Query, workspace: Path) -> Response:
    config_path = _required_workspace_path(query, "config", workspace)
    return Response.json(service.config_view(config_path))


def _handle_precheck(query: Query, workspace: Path) -> Response:
    return Response.json(service.precheck(_required_workspace_path(query, "config", workspace)))


def _handle_list_experiments(query: Query, workspace: Path) -> Response:
    database = _required_workspace_path(query, "database", workspace)
    return Response.json({"experiments": service.list_experiments(database)})


def _handle_experiment_detail(params: Params, query: Query, workspace: Path) -> Response:
    database = _required_workspace_path(query, "database", workspace)
    detail = service.experiment_detail(database, params["experiment_id"])
    return Response.json(detail)


def _handle_case_results(params: Params, query: Query, workspace: Path) -> Response:
    database = _required_workspace_path(query, "database", workspace)
    page = service.case_results(
        database,
        params["experiment_id"],
        config_id=_optional_str(query, "config_id"),
        status=_optional_str(query, "status"),
        limit=int(_optional_str(query, "limit") or 50),
        offset=int(_optional_str(query, "offset") or 0),
    )
    return Response.json(page)


def _handle_report(environ: WSGIEnviron, params: Params, workspace: Path) -> Response:
    # Writes report files: POST-only, same-origin JSON, workspace-confined.
    body = _read_json_body(environ)
    database = _resolve_in_workspace(workspace, _required_key(body, "database"))
    output = _resolve_in_workspace(workspace, _required_key(body, "output"))
    badge = body.get("badge")
    payload = service.generate_report(
        database,
        params["experiment_id"],
        output,
        badge=badge,
        baseline=body.get("baseline"),
    )
    return Response.json(payload)


def _handle_compare(environ: WSGIEnviron, workspace: Path) -> Response:
    # Writes a paired comparison report: POST-only, same-origin JSON,
    # workspace-confined.
    body = _read_json_body(environ)
    database = _resolve_in_workspace(workspace, _required_key(body, "database"))
    output = _resolve_in_workspace(workspace, _required_key(body, "output"))
    payload = service.generate_pair_report(
        database,
        _required_key(body, "baseline"),
        _required_key(body, "candidate"),
        _required_key(body, "baseline_config"),
        _required_key(body, "candidate_config"),
        output,
    )
    return Response.json(payload)


def _handle_download(query: Query, workspace: Path) -> Response:
    output_dir = _required_workspace_path(query, "output", workspace)
    file_path = service.artifact_download_path(output_dir, _required_str(query, "file"))
    return Response.file(file_path)


def _handle_start(environ: WSGIEnviron, manager: service.RunManager, workspace: Path) -> Response:
    body = _read_json_body(environ)
    config_path = _resolve_in_workspace(workspace, _required_key(body, "config"))
    token = manager.start_run(
        config_path,
        confirm_live_run=bool(body.get("confirm_live_run", False)),
    )
    return Response.json({"token": token})


def _handle_resume(
    environ: WSGIEnviron, params: Params, manager: service.RunManager, workspace: Path
) -> Response:
    body = _read_json_body(environ)
    config_path = _resolve_in_workspace(workspace, _required_key(body, "config"))
    token = manager.resume_run(
        config_path,
        params["experiment_id"],
        confirm_live_run=bool(body.get("confirm_live_run", False)),
        retry_unknown=bool(body.get("retry_unknown", False)),
    )
    return Response.json({"token": token})


def _handle_cancel(environ: WSGIEnviron, manager: service.RunManager, workspace: Path) -> Response:
    body = _read_json_body(environ)
    database_path = _resolve_in_workspace(workspace, _required_key(body, "database"))
    payload = service.request_cancel(database_path, body["experiment_id"])
    return Response.json(payload)


def _handle_run_status(params: Params, manager: service.RunManager) -> Response:
    try:
        state = manager.get(params["token"])
    except KeyError:
        return Response.error("404 Not Found", f"unknown run token: {params['token']}")
    return Response.json(
        {
            "status": state.status,
            "experiment_id": state.experiment_id,
            "error": state.error,
            "summary": state.summary,
        }
    )


def _resolve_in_workspace(workspace: Path, raw: str) -> Path:
    """Resolve ``raw`` against ``workspace`` and refuse it if it escapes."""

    raw_path = Path(raw)
    candidate = (workspace / raw_path if not raw_path.is_absolute() else raw_path).resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise PathEscapesWorkspace(f"path escapes workspace {workspace}: {raw}")
    return candidate


def _required_workspace_path(query: Query, key: str, workspace: Path) -> Path:
    return _resolve_in_workspace(workspace, _required_str(query, key))


def _required_str(query: Query, key: str) -> str:
    values = query.get(key)
    if not values:
        raise ValueError(f"missing required query parameter: {key}")
    return values[0]


def _optional_str(query: Query, key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _required_key(body: dict[str, Any], key: str) -> Any:
    if key not in body:
        raise ValueError(f"missing required field: {key}")
    return body[key]


def _read_json_body(environ: WSGIEnviron) -> dict[str, Any]:
    length = int(environ.get("CONTENT_LENGTH") or 0)
    raw = environ["wsgi.input"].read(length) if length else b"{}"
    payload = json.loads(raw or b"{}")
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def serve(
    *, host: str, port: int, allow_live: bool = False, workspace: Path | str | None = None
) -> WSGIServer:
    """Start a real HTTP server. Caller owns the returned server's lifecycle:
    run ``server.serve_forever()`` and always pair it with ``server.shutdown()``
    plus ``server_close()`` (a ``with`` block, or a ``try/finally``)."""

    return make_server(host, port, create_app(allow_live=allow_live, workspace=workspace))
