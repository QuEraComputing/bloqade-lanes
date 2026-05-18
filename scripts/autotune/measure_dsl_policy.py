"""Measure command for autotune: run the DSL policy and emit aggregated metrics.

Invokes `python -m benchmarks.cli` with the `dsl_autotune` strategy against
$BLOQADE_DSL_KERNELS (default: the full 9-kernel Squin suite), then reads
the resulting CSV and prints `AUTOTUNE_METRIC <name> <value>` lines to
stdout for autotune's RegexAdaptor to extract.

Primary metric is `total_events` — sum of `move_count_events` across kernels,
where each event is one parallel move timestep on the architecture. This is
the meaningful "solution length" measure (fewer timesteps = better), not the
total individual lane moves.

Environment:
  BLOQADE_DSL_POLICY    — relative path to the .star policy under evaluation
                          (default: policies/autotune/candidate.star).
  BLOQADE_DSL_KERNELS   — comma-separated kernel case_ids (default: full 9
                          kernels).
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_KERNELS = (
    "ghz_4,ghz_6,adder_4,steane_logical_5,"
    "qpe_9,"
    "adder_64,bv_70,steane_physical_35,trotter_rand_35"
)

# Penalty for a kernel that failed to compile/solve. Must exceed the largest
# plausible single-kernel `move_count_events` so that "solve poorly" always
# beats "skip the hard kernel" in the scorer's gradient. rust_entropy_5
# peaks at 1592 events on `adder_64`; 2000 keeps a margin.
FAIL_EVENTS_PENALTY = 2000.0


def _to_float(s: str) -> float:
    return float(s) if s else 0.0


def _emit(name: str, value: float) -> None:
    sys.stdout.write(f"AUTOTUNE_METRIC {name} {value}\n")


def main() -> int:
    kernels = os.environ.get("BLOQADE_DSL_KERNELS", DEFAULT_KERNELS)
    policy = os.environ.get("BLOQADE_DSL_POLICY", "policies/autotune/candidate.star")

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "dsl_run.csv"
        env = os.environ.copy()
        env["BLOQADE_DSL_POLICY"] = policy

        cmd = [
            sys.executable,
            "-m",
            "benchmarks.cli",
            "--cases",
            kernels,
            "--strategies",
            "dsl_autotune",
            "--architecture",
            "physical",
            "--repeats",
            "1",
            "--output",
            str(csv_path),
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            sys.stderr.write(f"benchmarks.cli exited with {result.returncode}\n")
            return result.returncode

        if not csv_path.exists():
            sys.stderr.write(f"expected CSV at {csv_path}, not produced\n")
            return 1

        with csv_path.open() as f:
            rows = list(csv.DictReader(f))

    if not rows:
        sys.stderr.write("measure_dsl_policy: empty CSV — no benchmark rows\n")
        return 1

    total_events = 0.0
    solved_events_sum = 0.0
    solved_count = 0
    total_nodes = 0.0
    total_wall = 0.0

    for row in rows:
        ok = row.get("success", "").strip().lower() == "true"
        events = _to_float(row.get("move_count_events", ""))
        nodes = _to_float(row.get("nodes_explored", ""))
        wall = _to_float(row.get("wall_time_ms", ""))

        if ok:
            solved_count += 1
            solved_events_sum += events
            total_events += events
        else:
            total_events += FAIL_EVENTS_PENALTY
        total_nodes += nodes
        total_wall += wall

    num_cases = len(rows)
    avg_events_solved = solved_events_sum / solved_count if solved_count else 9999.0
    success_rate = solved_count / num_cases

    _emit("total_events", total_events)
    _emit("success_rate", success_rate)
    _emit("avg_events_solved", avg_events_solved)
    _emit("total_nodes_explored", total_nodes)
    _emit("total_wall_time_ms", total_wall)
    _emit("num_cases", float(num_cases))
    _emit("num_solved", float(solved_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
