"""Detect whether the process that owns a running experiment still exists.

An owner is identified by host, PID, and a start marker derived from the
operating system's process start time, so a recycled PID is never mistaken
for the original owner. Owners on another host, or on platforms without a
start-time probe, are judged by a heartbeat lease instead.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

ProcessState = Literal["alive", "dead", "unknown"]
DEFAULT_LEASE_TTL_SECONDS = 60.0
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class ProcessOwner:
    """The process that holds an experiment's execution lease."""

    pid: int
    start_marker: str | None
    host: str


def current_owner(pid: int | None = None) -> ProcessOwner:
    """Describe a process on this host, defaulting to the current process."""

    target = os.getpid() if pid is None else pid
    return ProcessOwner(
        pid=target, start_marker=process_start_marker(target), host=socket.gethostname()
    )


def probe_process(pid: int, start_marker: str | None) -> ProcessState:
    """Report whether a local PID still runs the process that recorded the marker."""

    if not start_marker or not _start_markers_supported():
        return "unknown"
    marker = process_start_marker(pid)
    if marker is None:
        return "dead"
    return "alive" if marker == start_marker else "dead"


def process_start_marker(pid: int) -> str | None:
    """Return an opaque process start-time marker, or None if it is not running."""

    if sys.platform == "win32":
        return _windows_start_marker(pid)
    if sys.platform.startswith("linux"):
        return _linux_start_marker(pid)
    return None


def _start_markers_supported() -> bool:
    return sys.platform == "win32" or sys.platform.startswith("linux")


def _linux_start_marker(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8")
    except OSError:
        return None
    # Fields after the parenthesised command name start at field 3 (state);
    # starttime is field 22.
    fields = stat.rsplit(")", maxsplit=1)[-1].split()
    if len(fields) < 20 or fields[0] in {"Z", "X"}:
        return None
    return f"linux:{boot_id.strip()}:{fields[19]}"


def _windows_start_marker(pid: int) -> str | None:
    if sys.platform != "win32":  # pragma: no cover - narrows the platform for mypy
        return None
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return None
        if exit_code.value != still_active:
            return None
        creation = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        started = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return f"win32:{started}"
    finally:
        kernel32.CloseHandle(handle)


def utc_now() -> datetime:
    return datetime.now(UTC)


class LeaseHeartbeat:
    """Renew an experiment lease from a background thread while work runs."""

    def __init__(
        self,
        database_path: str | Path,
        experiment_id: str,
        lease_token: str,
        *,
        interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.database_path = Path(database_path)
        self.experiment_id = experiment_id
        self.lease_token = lease_token
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="experiment-lease-heartbeat", daemon=True
        )

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            while not self._stop.wait(self.interval_seconds):
                try:
                    with connection:
                        connection.execute(
                            """
                            UPDATE experiments SET heartbeat_at = ?
                            WHERE id = ? AND lease_token = ? AND status = 'running'
                            """,
                            (
                                utc_now().isoformat(),
                                self.experiment_id,
                                self.lease_token,
                            ),
                        )
                except sqlite3.OperationalError:
                    # A busy database only delays this beat; the next one retries.
                    continue
        finally:
            connection.close()
