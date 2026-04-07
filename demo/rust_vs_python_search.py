"""Compare Python and Rust move synthesis on circuits of varying size.

Compiles each circuit through the full pipeline using both backends and
compares wall time and solution quality (number of move steps, total lanes).

- Python: entropy-guided DFS + HeuristicMoveGenerator
- Rust:   A* search + HeuristicExpander (precomputed BFS distances)

Usage:
    uv run python demo/rust_vs_python_search.py
"""

import time

from kirin import ir
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.compile import compile_squin_to_move
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import PhysicalPlacementStrategy


# ── Circuits ─────────────────────────────────────────────────────────


@squin.kernel(typeinfer=True, fold=True)
def bell_pair():
    """2 qubits, 1 CZ."""
    q = qubit.qalloc(2)
    squin.h(q[0])
    squin.cz(q[0], q[1])


@squin.kernel(typeinfer=True, fold=True)
def ghz_4():
    """4 qubits, 3 CZ gates."""
    q = qubit.qalloc(4)
    squin.h(q[0])
    squin.cz(q[0], q[1])
    squin.cz(q[0], q[2])
    squin.cz(q[2], q[3])


@squin.kernel(typeinfer=True, fold=True)
def ghz_6():
    """6 qubits, 5 CZ gates (broadcast)."""
    q = qubit.qalloc(6)
    squin.h(q[0])
    squin.cz(q[0], q[1])
    squin.broadcast.cz(ilist.IList([q[0], q[1]]), ilist.IList([q[2], q[3]]))
    squin.broadcast.cz(ilist.IList([q[2], q[3]]), ilist.IList([q[4], q[5]]))


# ── Helpers ──────────────────────────────────────────────────────────

arch_spec = get_physical_arch_spec()
layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(arch_spec=arch_spec)


def count_moves(mt: ir.Method) -> tuple[int, int]:
    """Count (move_events, total_lanes_moved) in a compiled method."""
    events = 0
    lanes = 0
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, move.Move):
            events += 1
            lanes += len(stmt.lanes)
    return events, lanes


def compile_with(kernel, traversal, max_expansions=500):
    """Compile a kernel and return (time_ms, move_events, total_lanes, success)."""
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec, traversal=traversal, max_expansions=max_expansions
    )
    t0 = time.perf_counter()
    mt = compile_squin_to_move(
        kernel,
        layout_heuristic=layout_heuristic,
        placement_strategy=strategy,
        no_raise=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if mt is None:
        return elapsed, 0, 0, False

    events, lanes = count_moves(mt)
    return elapsed, events, lanes, True


# ── Main ─────────────────────────────────────────────────────────────

circuits = [
    ("Bell pair (2q, 1 CZ)", bell_pair),
    ("GHZ-4 (4q, 3 CZ)", ghz_4),
    ("GHZ-6 (6q, 5 CZ)", ghz_6),
]

print("=" * 72)
print("  Python vs Rust Move Synthesis")
print("=" * 72)
print()
print("  Python: entropy-guided DFS + HeuristicMoveGenerator")
print("  Rust:   A* search + HeuristicExpander (precomputed BFS)")
print()

header = f"  {'Circuit':<25s} {'Backend':>15s} {'Time':>8s} {'Moves':>6s} {'Lanes':>6s}"
print(header)
print("  " + "-" * (len(header) - 2))

for label, kernel in circuits:
    results = []
    for name, traversal in [("Python", "entropy"), ("Rust A*", "rust"), ("Rust DFS", "rust-dfs")]:
        ms, events, lanes, ok = compile_with(kernel, traversal)
        status = "" if ok else " FAILED"
        results.append((name, ms, events, lanes, ok))
        print(
            f"  {label:<25s} {name:>15s} {ms:7.0f}ms {events:6d} {lanes:6d}{status}"
        )
        label = ""  # only print circuit name on first line

    # Speedup: compare Python vs fastest Rust
    py = results[0]
    rust_results = [r for r in results[1:] if r[4]]
    if py[4] and rust_results:
        best_rust = min(rust_results, key=lambda r: r[1])
        speedup = py[1] / best_rust[1]
        move_diff = best_rust[2] - py[2]
        sign = "+" if move_diff > 0 else ""
        print(
            f"  {'':25s} {'':>15s} "
            f"  {speedup:.1f}x   {sign}{move_diff} moves"
        )
    print()

print("=" * 72)
print("  'Moves' = number of move instructions (parallel move steps)")
print("  'Lanes' = total individual atom transports across all moves")
print("=" * 72)
