# ruff: noqa: E402
"""Measure command for autotune: run the DSL policy and emit aggregated metrics.

Calls `bloqade.lanes.compile.compile_squin_to_move` directly with a
`PhysicalPlacementStrategy(traversal=RustPlacementTraversal(policy_path=...))`
on each requested benchmark kernel. Prints `AUTOTUNE_METRIC <name> <value>`
lines to stdout for autotune's RegexAdaptor to extract, and per-case
diagnostics to stderr.

Deliberately bypasses `python -m benchmarks.cli` (and its
`squin_to_move`/`_apply_architecture_mode` plumbing) because that path
overrides each kernel's `logical_initialize` flag based on the
`--architecture` argument, which is a logical-qubit concept that has no
bearing on physical-arch move-policy search. `compile_squin_to_move` is
the physical-only entry point (wraps `PhysicalPipeline` with the physical
arch spec) and has no `logical_initialize` parameter.

Primary solution-quality metric is `move_layers` — sum of `move.Move` statement counts
across kernels (one statement = one parallel move timestep on the arch).

Environment:
  BLOQADE_DSL_POLICY    — relative path to the .star policy under evaluation
                          (default: policies/autotune/candidate.star).
  BLOQADE_DSL_KERNELS   — comma-separated kernel case_ids (default:
                          steane_physical_35).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Keep third-party imports from trying to write under the user's home
# directory when autotune runs in a non-interactive sandbox.
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".autotune/matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from benchmarks.kernels import select_benchmark_cases

from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.compile import compile_squin_to_move
from bloqade.lanes.dialects import move as move_dialect, place as place_dialect
from bloqade.lanes.heuristics.physical import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)

DEFAULT_KERNELS = "steane_physical_35"

# Penalty for a kernel that failed to compile/solve. Must exceed the largest
# plausible single-kernel `move_count_events` so that "solve poorly" always
# beats "skip the hard kernel" in the scorer's gradient. rust_entropy_5
# peaks at 1592 events on `adder_64`; 2000 keeps a margin.
FAIL_EVENTS_PENALTY = 2000.0


def _emit(name: str, value: float) -> None:
    sys.stdout.write(f"AUTOTUNE_METRIC {name} {value}\n")


def _count_move_events(mt: object) -> int:
    return sum(
        1
        for stmt in mt.callable_region.walk()  # type: ignore[attr-defined]
        if isinstance(stmt, move_dialect.Move)
    )


def _build_strategy(
    policy_path: str,
) -> tuple[PalindromePlacementStrategy, PhysicalPlacementStrategy]:
    """Returns (wrapped, inner). The wrapped strategy is what the pipeline
    consumes (matches the default `PhysicalPipeline` wraps PhysicalPlacementStrategy
    in PalindromePlacementStrategy). The inner is returned separately so the
    caller can read `rust_nodes_expanded_total` after the solve."""
    inner = PhysicalPlacementStrategy(
        arch_spec=get_physical_arch_spec(),
        traversal=RustPlacementTraversal(
            policy_path=policy_path,
            max_expansions=1000,
            timeout_s=30.0,
        ),
    )
    return PalindromePlacementStrategy(inner=inner), inner


def _has_unlowered_place_cz(mt: object) -> bool:
    """True iff any `place.CZ` statements remain in the compiled IR.

    `compile_squin_to_move` with `no_raise=True` returns silently even when
    a placement strategy fails to solve every CZ stage — the leftover
    `place.CZ` statements are the signal that the solve was incomplete.
    """
    return any(
        isinstance(stmt, place_dialect.CZ)
        for stmt in mt.callable_region.walk()  # type: ignore[attr-defined]
    )


def main() -> int:
    kernels_csv = os.environ.get("BLOQADE_DSL_KERNELS", DEFAULT_KERNELS)
    policy = os.environ.get("BLOQADE_DSL_POLICY", "policies/autotune/candidate.star")
    case_ids = {c.strip() for c in kernels_csv.split(",") if c.strip()}

    # Archive the policy file content to stderr so autotune's iteration
    # `measure_output/dsl_benchmark.stderr.txt` preserves what the
    # implementer actually wrote.
    sys.stderr.write(f"---POLICY FILE: {policy}---\n")
    try:
        sys.stderr.write(Path(policy).read_text())
    except OSError as exc:
        sys.stderr.write(f"[could not read policy: {exc}]\n")
    sys.stderr.write("\n---END POLICY FILE---\n")

    try:
        cases = select_benchmark_cases(case_ids)
    except ValueError as exc:
        sys.stderr.write(f"measure_dsl_policy: bad kernel filter: {exc}\n")
        return 1

    total_events = 0.0
    solved_events_sum = 0.0
    solved_count = 0
    total_nodes = 0.0
    total_wall = 0.0
    num_cases = len(cases)

    for case in cases:
        wrapped_strategy, inner_strategy = _build_strategy(policy)
        layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut()

        start = time.perf_counter()
        err: str | None = None
        events = 0
        unlowered = True
        try:
            mt = compile_squin_to_move(
                case.kernel,
                no_raise=True,
                layout_heuristic=layout_heuristic,
                placement_strategy=wrapped_strategy,
            )
            unlowered = _has_unlowered_place_cz(mt)
            events = _count_move_events(mt)
        except Exception as exc:  # broad on purpose — surface unexpected errors
            err = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        nodes = inner_strategy.rust_nodes_expanded_total
        ok = (err is None) and (not unlowered)

        if ok:
            sys.stderr.write(
                f"AUTOTUNE_NOTE {case.case_id} ok=True events={events} "
                f"nodes={nodes} wall_ms={elapsed_ms:.3f}\n"
            )
            solved_count += 1
            solved_events_sum += float(events)
            total_events += float(events)
        else:
            reason = (
                err
                if err is not None
                else (
                    "place.CZ statements remain (policy failed to lower every CZ "
                    "stage; this usually means at least one CZ pair was not placed "
                    "on compatible neighboring physical sites)"
                )
            )
            sys.stderr.write(
                f"AUTOTUNE_NOTE {case.case_id} ok=False nodes={nodes} "
                f"wall_ms={elapsed_ms:.3f} reason={reason}\n"
            )
            if err is not None:
                sys.stderr.write(traceback.format_exc())
            total_events += FAIL_EVENTS_PENALTY
        total_nodes += float(nodes)
        total_wall += elapsed_ms

    avg_events_solved = solved_events_sum / solved_count if solved_count else 9999.0
    success_rate = solved_count / num_cases if num_cases else 0.0

    _emit("total_events", total_events)
    _emit("move_layers", total_events)
    _emit("success_rate", success_rate)
    _emit("avg_events_solved", avg_events_solved)
    _emit("total_nodes_explored", total_nodes)
    _emit("total_wall_time_ms", total_wall)
    _emit("num_cases", float(num_cases))
    _emit("num_solved", float(solved_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
