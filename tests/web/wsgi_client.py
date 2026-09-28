"""A minimal in-process WSGI test client: no sockets, no network."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

from rag_quality_lab.web.app import WSGIApp


@dataclass
class WSGIResponse:
    status: str
    body: bytes

    @property
    def status_code(self) -> int:
        return int(self.status.split(" ", 1)[0])

    def json(self) -> Any:
        return json.loads(self.body)


def call(
    app: WSGIApp,
    method: str,
    path: str,
    *,
    query: str = "",
    json_body: dict[str, Any] | None = None,
) -> WSGIResponse:
    payload = json.dumps(json_body).encode("utf-8") if json_body is not None else b""
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": str(len(payload)),
        "wsgi.input": io.BytesIO(payload),
    }
    captured: dict[str, Any] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(app(environ, start_response))
    return WSGIResponse(status=captured["status"], body=body)
