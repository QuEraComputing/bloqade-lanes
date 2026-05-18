"""Adaptor script for autotune: aggregate benchmark CSV rows into JSON metrics.

Reads the CSV emitted by `measure_dsl_policy.py` from stdin (the measure
command's stdout, piped via autotune's ScriptAdaptor) and prints a flat JSON
object of `{metric_name: f64}` to stdout for autotune to score.

Each input row corresponds to one benchmark case run under the `dsl_autotune`
strategy. Failed rows (success=False) are penalised with `FAIL_LANES_PENALTY`
move lanes so the LLM has a gradient toward "make more kernels succeed."

Output metrics:
  total_lanes           — sum of move_count_lanes (failed rows count as penalty)
  avg_lanes_solved      — mean move_count_lanes over successful rows only
                          (NaN sentinel `9999.0` if no rows succeeded)
  success_rate          — fraction of rows with success=True
  total_nodes_explored  — sum of nodes_explored across all rows
  total_wall_time_ms    — sum of wall_time_ms across all rows
  num_cases             — total row count (sanity)
  num_solved            — count of success=True rows
"""

from __future__ import annotations

import csv
import json
import sys

# Penalty for a kernel that failed to compile/solve. Must exceed the largest
# plausible single-kernel solution length so that "solve poorly" always beats
# "don't solve at all" — otherwise the LLM gets a perverse gradient toward
# skipping the hard kernels. `rust_entropy_5` peaks at 1660 lanes on
# `adder_64`; 2500 keeps a comfortable margin.
FAIL_LANES_PENALTY = 2500.0


def _to_float(s: str) -> float:
    return float(s) if s else 0.0


def main() -> int:
    reader = csv.DictReader(sys.stdin)
    rows = list(reader)
    if not rows:
        sys.stderr.write("parse_csv_metrics: empty CSV on stdin\n")
        return 1

    total_lanes = 0.0
    solved_lanes_sum = 0.0
    solved_count = 0
    total_nodes = 0.0
    total_wall = 0.0

    for row in rows:
        ok = row.get("success", "").strip().lower() == "true"
        lanes = _to_float(row.get("move_count_lanes", ""))
        nodes = _to_float(row.get("nodes_explored", ""))
        wall = _to_float(row.get("wall_time_ms", ""))

        if ok:
            solved_count += 1
            solved_lanes_sum += lanes
            total_lanes += lanes
        else:
            total_lanes += FAIL_LANES_PENALTY
        total_nodes += nodes
        total_wall += wall

    num_cases = len(rows)
    metrics = {
        "total_lanes": total_lanes,
        "avg_lanes_solved": (
            solved_lanes_sum / solved_count if solved_count else 9999.0
        ),
        "success_rate": solved_count / num_cases,
        "total_nodes_explored": total_nodes,
        "total_wall_time_ms": total_wall,
        "num_cases": float(num_cases),
        "num_solved": float(solved_count),
    }
    json.dump(metrics, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
