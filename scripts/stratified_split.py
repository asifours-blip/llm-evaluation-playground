"""Derive a dataset version with a seeded, stratified dev/holdout split.

Strata are (first tag, difficulty). Each tag contributes round(n_tag * fraction)
holdout cases, spread over its difficulty strata by largest remainder; cases
whose review is not approved are never drawn into the holdout. Re-running with
the same inputs and seed reproduces the same split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DIFFICULTY_ORDER = ("easy", "medium", "hard")


def allocate(counts: dict[str, int], target: int) -> dict[str, int]:
    """Split ``target`` across strata proportionally, by largest remainder."""

    total = sum(counts.values())
    if total == 0:
        return {key: 0 for key in counts}
    exact = {key: count * target / total for key, count in counts.items()}
    quota = {key: int(value) for key, value in exact.items()}
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - quota[key]), DIFFICULTY_ORDER.index(key)),
    )
    for key in order[: target - sum(quota.values())]:
        quota[key] += 1
    return quota


def split_cases(
    cases: list[dict[str, Any]], rejected: set[str], fraction: float, seed: int
) -> dict[str, str]:
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_tag[case["tags"][0]].append(case)

    splits = {case["id"]: "dev" for case in cases}
    for tag, tag_cases in sorted(by_tag.items()):
        target = round(len(tag_cases) * fraction)
        eligible: dict[str, list[str]] = defaultdict(list)
        for case in tag_cases:
            if case["id"] not in rejected:
                eligible[case["difficulty"]].append(case["id"])
        quota = allocate({key: len(ids) for key, ids in eligible.items()}, target)
        for difficulty, ids in sorted(eligible.items()):
            ordered = sorted(ids)
            random.Random(f"{seed}:{tag}:{difficulty}").shuffle(ordered)
            for case_id in ordered[: quota[difficulty]]:
                splits[case_id] = "holdout"
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float, default=1 / 3)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", required=True, help="ISO 8601 timestamp")
    parser.add_argument("--reject", nargs="*", default=[], help="case IDs whose review failed")
    parser.add_argument("--description", required=True)
    args = parser.parse_args()

    dataset = json.loads(args.source.read_text(encoding="utf-8"))
    rejected = set(args.reject)
    unknown = rejected - {case["id"] for case in dataset["cases"]}
    if unknown:
        parser.error(f"unknown case IDs: {sorted(unknown)}")
    reviewed_at = datetime.fromisoformat(args.reviewed_at).astimezone(UTC)

    splits = split_cases(dataset["cases"], rejected, args.fraction, args.seed)
    for case in dataset["cases"]:
        case["split"] = splits[case["id"]]
        case["review"] = {
            "status": "rejected" if case["id"] in rejected else "approved",
            "reviewer": args.reviewer,
            "reviewed_at": reviewed_at.isoformat().replace("+00:00", "Z"),
        }
    dataset["version"] = args.version
    dataset["description"] = args.description
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
