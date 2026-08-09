#!/usr/bin/env python3
"""Append a completed ``perf stat`` result to a rank hardware ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .samplers import PerfStatReader


def append_perf_stat(
    perf_path: Path, output_path: Path, *, run_id: str, rank: int
) -> dict[str, Any]:
    record = PerfStatReader(perf_path).snapshot()
    record.update({"run_id": run_id, "rank": int(rank)})
    payload = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-stat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    record = append_perf_stat(
        args.perf_stat, args.output, run_id=args.run_id, rank=args.rank
    )
    print(json.dumps({"status": record["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
