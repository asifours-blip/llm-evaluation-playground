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


route("GET", r"/", "index")
route("GET", r"/api/dataset", "dataset")
route("GET", r"/api/config", "config")
route("GET", r"/api/precheck", "precheck")
route("GET", r"/api/experiments", "list_experiments")
route("GET", r"/api/experiments/(?P<experiment_id>[^/]+)", "experiment_detail")
route("GET", r"/api/experiments/(?P<experiment_id>[^/]+)/results", "case_results")
route("GET", r"/api/experiments/(?P<experiment_id>[^/]+)/report", "report")
route("POST", r"/api/experiments/start", "start")
route("POST", r"/api/experiments/(?P<experiment_id>[^/]+)/resume", "resume")
route("POST", r"/api/experiments/(?P<experiment_id>[^/]+)/cancel", "cancel")
route("GET", r"/api/runs/(?P<token>[^/]+)", "run_status")
route("GET", r"/api/compare", "compare")
route("GET", r"/api/artifacts/download", "download")


WSGIApp = Callable[[WSGIEnviron, StartResponse], Iterable[bytes]]


def create_app(*, allow_live: bool = False) -> WSGIApp:
    """Build the WSGI application. ``allow_live=False`` (the default) refuses
    every live run regardless of what a request asks for."""

    manager = service.RunManager(allow_live=allow_live)
    handlers: dict[str, Handler] = {
        "index": _handle_index,
        "dataset": _handle_dataset,
        "config": _handle_config,
        "precheck": _handle_precheck,
        "list_experiments": _handle_list_experiments,
        "experiment_detail": _handle_experiment_detail,
        "case_results": _handle_case_results,
        "report": _handle_report,
        "compare": _handle_compare,
        "download": _handle_download,
        "start": lambda environ, params, query: _handle_start(environ, manager),
        "resume": lambda environ, params, query: _handle_resume(environ, params, manager),
        "cancel": lambda environ, params, query: _handle_cancel(environ, manager),
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
    for route_method, pattern, name in _ROUTES:
        if route_method != method:
            continue
        match = pattern.match(path)
        if match is None:
            continue
        try:
            return handlers[name](environ, match.groupdict(), query)
        except KeyError as error:
            return Response.error("404 Not Found", str(error))
        except FileNotFoundError as error:
            return Response.error("404 Not Found", str(error))
        except service.PathEscapesArtifactDir as error:
            return Response.error("400 Bad Request", str(error))
        except service.LiveRunNotAllowed as error:
            return Response.error("403 Forbidden", str(error))
        except (ValueError, PermissionError) as error:
            return Response.error("400 Bad Request", str(error))
    return Response.error("404 Not Found", f"no route for {method} {path}")


def _handle_index(environ: WSGIEnviron, params: Params, query: Query) -> Response:
    return Response.file(STATIC_DIR / "index.html")


def _handle_dataset(environ: WSGIEnviron, params: Params, query: Query) -> Response:
    return Response.json(service.dataset_view(_required_path(query, "dataset")))


def _handle_config(environ: WSGIEnviron, params: Params, query: Query) -> Response:
    return Response.json(service.config_view(_required_path(query, "config")))


def _handle_precheck(environ: WSGIEnviron, params: Params, query: Query) -> Response:
    return Response.json(service.precheck(_required_path(query, "config")))


def _handle_list_experiments(
    environ: WSGIEnviron, params: Params, query: Query
) -> Response:
    return Response.json(
        {"experiments": service.list_experiments(_required_path(query, "database"))}
    )


def _handle_experiment_detail(
    environ: WSGIEnviron, params: Params, query: Query
) -> Response:
    detail = service.experiment_detail(
        _required_path(query, "database"), params["experiment_id"]
    )
    return Response.json(detail)


def _handle_case_results(
    environ: WSGIEnviron, params: Params, query: Query
) -> Response:
    page = service.case_results(
        _required_path(query, "database"),
        params["experiment_id"],
        config_id=_optional_str(query, "config_id"),
        status=_optional_str(query, "status"),
        limit=int(_optional_str(query, "limit") or 50),
        offset=int(_optional_str(query, "offset") or 0),
    )
    return Response.json(page)


def _handle_report(environ: WSGIEnviron, params: Params, query: Query) -> Response:
    badge = _optional_str(query, "badge")
    payload = service.generate_report(
        _required_path(query, "database"),
        params["experiment_id"],
        _required_path(query, "output"),
        badge=badge,  # type: ignore[arg-type]
        baseline=_optional_str(query, "baseline"),
    )
    return Response.json(payload)


def _handle_compare(environ: WSGIEnviron, params: Params, query: Query) -> Response:
    payload = service.generate_pair_report(
        _required_path(query, "database"),
        _required_str(query, "baseline"),
        _required_str(query, "candidate"),
        _required_str(query, "baseline_config"),
        _required_str(query, "candidate_config"),
        _required_path(query, "output"),
    )
    return Response.json(payload)


def _handle_download(environ: WSGIEnviron, params: Params, query: Query) -> Response:
    output_dir = _required_path(query, "output")
    file_path = service.artifact_download_path(output_dir, _required_str(query, "file"))
    return Response.file(file_path)


def _handle_start(environ: WSGIEnviron, manager: service.RunManager) -> Response:
    body = _read_json_body(environ)
    token = manager.start_run(
        Path(_required_key(body, "config")),
        confirm_live_run=bool(body.get("confirm_live_run", False)),
    )
    return Response.json({"token": token})


def _handle_resume(
    environ: WSGIEnviron, params: dict[str, str], manager: service.RunManager
) -> Response:
    body = _read_json_body(environ)
    token = manager.resume_run(
        Path(_required_key(body, "config")),
        params["experiment_id"],
        confirm_live_run=bool(body.get("confirm_live_run", False)),
        retry_unknown=bool(body.get("retry_unknown", False)),
    )
    return Response.json({"token": token})


def _handle_cancel(environ: WSGIEnviron, manager: service.RunManager) -> Response:
    body = _read_json_body(environ)
    payload = service.request_cancel(Path(_required_key(body, "database")), body["experiment_id"])
    return Response.json(payload)


def _handle_run_status(params: dict[str, str], manager: service.RunManager) -> Response:
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


def _required_path(query: Query, key: str) -> Path:
    return Path(_required_str(query, key))


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


def serve(*, host: str, port: int, allow_live: bool = False) -> WSGIServer:
    """Start a real HTTP server. Caller owns the returned server's lifecycle:
    run ``server.serve_forever()`` and always pair it with ``server.shutdown()``
    plus ``server_close()`` (a ``with`` block, or a ``try/finally``)."""

    return make_server(host, port, create_app(allow_live=allow_live))
