#!/usr/bin/env python3
"""Summarize the phase-tagged nvidia-smi samples produced by run_smtgazer.sh."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


NUMERIC_FIELDS = (
    "gpu_util_pct",
    "memory_util_pct",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "power_limit_w",
    "temperature_c",
    "sm_clock_mhz",
)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(rows: list[dict]) -> dict:
    values = {
        field: [float(row[field]) for row in rows if row.get(field, "") != ""]
        for field in NUMERIC_FIELDS
    }
    timestamps = [parse_timestamp(row["timestamp_utc"]) for row in rows]
    energy_wh = 0.0
    for current, following in zip(rows, rows[1:]):
        delta = (
            parse_timestamp(following["timestamp_utc"])
            - parse_timestamp(current["timestamp_utc"])
        ).total_seconds()
        if 0 < delta <= 60 and current.get("power_draw_w"):
            energy_wh += float(current["power_draw_w"]) * delta / 3600

    utilization = values["gpu_util_pct"]
    memory_used = values["memory_used_mib"]
    power = values["power_draw_w"]
    return {
        "samples": len(rows),
        "duration_seconds": max(0.0, (max(timestamps) - min(timestamps)).total_seconds()),
        "gpu_util_avg_pct": mean(utilization),
        "gpu_util_p50_pct": percentile(utilization, 0.50),
        "gpu_util_p95_pct": percentile(utilization, 0.95),
        "gpu_util_max_pct": max(utilization),
        "gpu_active_samples_pct": 100 * sum(value >= 10 for value in utilization) / len(utilization),
        "memory_util_avg_pct": mean(values["memory_util_pct"]),
        "memory_used_avg_mib": mean(memory_used),
        "memory_used_peak_mib": max(memory_used),
        "power_draw_avg_w": mean(power),
        "power_draw_peak_w": max(power),
        "estimated_energy_wh": energy_wh,
        "temperature_peak_c": max(values["temperature_c"]),
        "sm_clock_avg_mhz": mean(values["sm_clock_mhz"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SMTgazer GPU samples")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"No GPU samples in {args.input}")

    phases = defaultdict(list)
    for row in rows:
        phases[row["phase"]].append(row)
    report = {
        "source": str(args.input.resolve()),
        "overall": summarize(rows),
        "phases": {phase: summarize(samples) for phase, samples in phases.items()},
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
