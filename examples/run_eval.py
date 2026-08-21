"""Compatibility wrapper for the deterministic offline evaluation workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_quality_lab.cli import main as cli_main  # noqa: E402


def main() -> int:
    """Delegate without preserving the superseded evaluator implementation."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock",
        action="store_true",
        help="required safety flag: this compatibility entry is offline-only",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "offline.yaml",
    )
    args = parser.parse_args()
    if not args.mock:
        parser.error("this compatibility entry requires --mock; use rag-quality for live runs")
    print("[MOCK] Delegating to the reproducible rag-quality workflow.")
    return cli_main(["run", "--config", str(args.config)])


if __name__ == "__main__":
    raise SystemExit(main())
