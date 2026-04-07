"""Demo: compile a Bell pair circuit with Python and Rust move synthesis.

The Rust A* solver with the exhaustive AOD rectangle generator currently
has a very high branching factor on the full physical architecture (many
possible rectangles per bus triplet). This means a single node expansion
is expensive, and the search may not complete within a reasonable budget.

This demo shows the Python entropy-guided search working, and the Rust
solver on a standalone problem (bypassing the full pipeline) to
demonstrate it works correctly.

Usage:
    uv run python demo/rust_vs_python_search.py
"""

import json
import time

from bloqade import qubit, squin
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.bytecode._native import MoveSolver
from bloqade.lanes.compile import compile_squin_to_move
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import PhysicalPlacementStrategy

# ── Part 1: Full pipeline with Python entropy search ─────────────────


@squin.kernel(typeinfer=True, fold=True)
def bell_pair():
    q = qubit.qalloc(2)
    squin.h(q[0])
    squin.cz(q[0], q[1])


arch_spec = get_physical_arch_spec()
layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(arch_spec=arch_spec)
strategy = PhysicalPlacementStrategy(arch_spec=arch_spec, traversal="entropy")

print("=" * 60)
print("  Part 1: Full pipeline (Python entropy search)")
print("=" * 60)
print()

t0 = time.perf_counter()
move_mt = compile_squin_to_move(
    bell_pair,
    layout_heuristic=layout_heuristic,
    placement_strategy=strategy,
    no_raise=True,
)
elapsed = (time.perf_counter() - t0) * 1000
print(f"  Bell pair compilation: {elapsed:.0f} ms")
print(f"  Success: {move_mt is not None}")
print()


# ── Part 2: Rust A* solver on a standalone placement problem ─────────

print("=" * 60)
print("  Part 2: Rust A* solver (standalone placement problems)")
print("=" * 60)
print()

# Use a small test architecture where the branching factor is manageable.
small_arch = json.dumps(
    {
        "version": "2.0",
        "geometry": {
            "sites_per_word": 10,
            "words": [
                {
                    "positions": {
                        "x_start": 1.0,
                        "y_start": 2.5,
                        "x_spacing": [2.0, 2.0, 2.0, 2.0],
                        "y_spacing": [2.5],
                    },
                    "site_indices": [
                        [0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [4, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [3, 1],
                        [4, 1],
                    ],
                },
                {
                    "positions": {
                        "x_start": 1.0,
                        "y_start": 12.5,
                        "x_spacing": [2.0, 2.0, 2.0, 2.0],
                        "y_spacing": [2.5],
                    },
                    "site_indices": [
                        [0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [4, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [3, 1],
                        [4, 1],
                    ],
                },
            ],
        },
        "buses": {
            "site_buses": [{"src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9]}],
            "word_buses": [{"src": [0], "dst": [1]}],
        },
        "words_with_site_buses": [0, 1],
        "sites_with_word_buses": [5, 6, 7, 8, 9],
        "zones": [{"words": [0, 1]}],
        "entangling_zones": [[[0, 1]]],
        "blockade_radius": 2.0,
        "measurement_mode_zones": [0],
    }
)

solver = MoveSolver(small_arch)

problems = [
    {
        "name": "Single qubit, 1 hop (site bus)",
        "initial": [(0, 0, 0)],
        "target": [(0, 0, 5)],
        "blocked": [],
    },
    {
        "name": "Single qubit, 2 hops (site + word bus)",
        "initial": [(0, 0, 0)],
        "target": [(0, 1, 5)],
        "blocked": [],
    },
    {
        "name": "Two qubits, parallel move",
        "initial": [(0, 0, 0), (1, 0, 1)],
        "target": [(0, 0, 5), (1, 0, 6)],
        "blocked": [],
    },
    {
        "name": "Single qubit, blocked destination",
        "initial": [(0, 0, 0)],
        "target": [(0, 0, 5)],
        "blocked": [(0, 5)],
    },
    {
        "name": "Cross-word with obstacle",
        "initial": [(0, 0, 5), (1, 0, 6)],
        "target": [(0, 1, 5), (1, 1, 6)],
        "blocked": [],
    },
]

for p in problems:
    t0 = time.perf_counter()
    result = solver.solve(p["initial"], p["target"], p["blocked"], max_expansions=1000)
    elapsed_us = (time.perf_counter() - t0) * 1_000_000

    if result is not None:
        print(f"  {p['name']}")
        print(
            f"    Steps: {len(result.move_layers)}, "
            f"Cost: {result.cost:.0f}, "
            f"Expanded: {result.nodes_expanded}, "
            f"Time: {elapsed_us:.0f} us"
        )
    else:
        print(f"  {p['name']}")
        print(f"    No solution found ({elapsed_us:.0f} us)")

print()
print("=" * 60)
print("  Note: The Rust A* exhaustive generator has a high branching")
print("  factor on the full physical architecture. A heuristic move")
print("  generator (like Python's HeuristicMoveGenerator) would be")
print("  needed for production use on large architectures.")
print("=" * 60)
