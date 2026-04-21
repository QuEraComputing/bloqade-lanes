"""Compile a small physical squin kernel with Default vs Congestion-aware
target generators, and report move-event and lane counts for each.

Run with:
  uv run python demo/congestion_aware_compile.py
  uv run python demo/congestion_aware_compile.py --visualize ghz_optimal
  uv run python demo/congestion_aware_compile.py --visualize steane
"""

from __future__ import annotations

import argparse

from benchmarks.kernels.large.steane_physical_35 import steane_physical_35
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.analysis.atom.analysis import AtomInterpreter
from bloqade.lanes.analysis.atom.lattice import AtomState
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.compile import (
    compile_squin_to_move,
    compile_squin_to_move_and_visualize,
    compile_squin_to_move_best,
)
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.physical.placement import PhysicalPlacementStrategy
from bloqade.lanes.heuristics.physical.target_generator import (
    AODClusterTargetGenerator,
    CongestionAwareTargetGenerator,
    DefaultTargetGenerator,
    TargetGeneratorABC,
)


@squin.kernel(typeinfer=True, fold=True)
def ghz_6():
    """6-qubit GHZ via log-depth doubling."""
    size = 6
    q0 = qubit.new()
    squin.h(q0)
    reg = ilist.IList([q0])
    for _ in range(size):
        current = len(reg)
        missing = size - current
        if missing > current:
            num_alloc = current
        else:
            num_alloc = missing
        if num_alloc > 0:
            new_qubits = qubit.qalloc(num_alloc)
            squin.broadcast.cx(reg[-num_alloc:], new_qubits)
            reg = reg + new_qubits


@squin.kernel(typeinfer=True, fold=True)
def ghz_optimal():
    """10-qubit GHZ using explicit broadcast-CZ layers with nested pairs.

    Each broadcast.cz is a multi-pair CZ stage — exactly the shape
    where congestion-aware direction selection can diverge from
    default's rule-based control-always-moves.
    """
    qs = qubit.qalloc(10)
    squin.broadcast.sqrt_y(qs)
    squin.z(qs[0])
    squin.cz(qs[0], qs[5])
    squin.broadcast.cz(ilist.IList([qs[0], qs[5]]), ilist.IList([qs[1], qs[6]]))
    squin.broadcast.cz(qs[:2] + qs[5:7], qs[2:4] + qs[7:9])
    squin.broadcast.cz(ilist.IList([qs[3], qs[8]]), ilist.IList([qs[4], qs[9]]))
    squin.broadcast.sqrt_y(qs)
    squin.sqrt_y_adj(qs[0])


def count_moves(move_mt) -> tuple[int, int]:
    events = 0
    lanes = 0
    for stmt in move_mt.callable_region.walk():
        if isinstance(stmt, move.Move):
            events += 1
            lanes += len(stmt.lanes)
    return events, lanes


def compile_with(kernel, generator: TargetGeneratorABC | None, label: str):
    arch_spec = get_arch_spec()
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec, target_generator=generator
    )
    move_mt = compile_squin_to_move(kernel, placement_strategy=strategy)
    events, lanes = count_moves(move_mt)
    print(f"{label:>22}:  move events = {events:4d}   total lanes = {lanes:5d}")


def compare_on(kernel, name: str) -> None:
    print(f"\nkernel: {name}   arch: physical Gemini")
    compile_with(kernel, DefaultTargetGenerator(), "DefaultTargetGenerator")
    compile_with(kernel, CongestionAwareTargetGenerator(), "CongestionAware")


def per_atom_move_counts(move_mt, arch_spec) -> dict[int, int]:
    """Walk the move program and return the final per-qid move count."""
    frame, _ = AtomInterpreter(move_mt.dialects, arch_spec=arch_spec).run(move_mt)
    last_state: AtomState | None = None
    for stmt in move_mt.callable_region.walk():
        for v in frame.get_values(stmt.results):
            if isinstance(v, AtomState):
                last_state = v
    assert last_state is not None, "no AtomState observed in move program"
    return dict(last_state.data.move_count)


def analyze_on(kernel, name: str) -> None:
    print(f"\nkernel: {name}   arch: physical Gemini")
    arch_spec = get_arch_spec()
    rows: list[tuple[str, dict[int, int]]] = []

    labelled_strategies: list[tuple[str, PhysicalPlacementStrategy]] = [
        (
            "Default",
            PhysicalPlacementStrategy(
                arch_spec=arch_spec, target_generator=DefaultTargetGenerator()
            ),
        ),
        (
            "CongestionAware",
            PhysicalPlacementStrategy(
                arch_spec=arch_spec, target_generator=CongestionAwareTargetGenerator()
            ),
        ),
        (
            "AODCluster",
            PhysicalPlacementStrategy(
                arch_spec=arch_spec, target_generator=AODClusterTargetGenerator()
            ),
        ),
    ]

    for label, strategy in labelled_strategies:
        move_mt = compile_squin_to_move(kernel, placement_strategy=strategy)
        events, lanes = count_moves(move_mt)
        counts = per_atom_move_counts(move_mt, arch_spec)
        total = sum(counts.values())
        mx = max(counts.values()) if counts else 0
        print(
            f"  {label:>15}: events={events:4d} lanes={lanes:4d} "
            f"per-atom total={total:4d} max={mx:3d}"
        )
        rows.append((label, counts))

    best_mt, winner = compile_squin_to_move_best(kernel, strategies=labelled_strategies)
    best_events, best_lanes = count_moves(best_mt)
    best_counts = per_atom_move_counts(best_mt, arch_spec)
    best_total = sum(best_counts.values())
    best_max = max(best_counts.values()) if best_counts else 0
    print(
        f"  {'Best(' + winner + ')':>15}: events={best_events:4d} lanes={best_lanes:4d} "
        f"per-atom total={best_total:4d} max={best_max:3d}"
    )
    rows.append((f"Best({winner})", best_counts))

    all_qids = sorted({q for _, counts in rows for q in counts.keys()})
    print("  per-atom move counts by qid:")
    header = f"    {'':>15}  " + "  ".join(f"q{q:>2}" for q in all_qids)
    print(header)
    for label, counts in rows:
        row = f"    {label:>15}  " + "  ".join(
            f"{counts.get(q, 0):>3}" for q in all_qids
        )
        print(row)
    for (a_label, a_counts), (b_label, b_counts) in zip(rows, rows[1:]):
        diffs = [q for q in all_qids if a_counts.get(q, 0) != b_counts.get(q, 0)]
        if diffs:
            print(f"  qids differing {a_label} vs {b_label}: {diffs}")
        else:
            print(f"  identical distributions {a_label} vs {b_label}")


def visualize(kernel, name: str) -> None:
    arch_spec = get_arch_spec()
    for gen, label in [
        (DefaultTargetGenerator(), "Default"),
        (CongestionAwareTargetGenerator(), "CongestionAware"),
        (AODClusterTargetGenerator(), "AODCluster"),
    ]:
        print(f"\n>>> {name} with {label} — close the debugger window to continue")
        strategy = PhysicalPlacementStrategy(arch_spec=arch_spec, target_generator=gen)
        compile_squin_to_move_and_visualize(
            kernel, placement_strategy=strategy, interactive=True
        )


KERNELS = {
    "ghz_6": (ghz_6, "ghz_6 (log-depth doubling)"),
    "ghz_optimal": (ghz_optimal, "ghz_optimal (broadcast-CZ layers)"),
    "steane": (steane_physical_35, "steane_physical_35 (3 broadcast-CZ stages)"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize",
        choices=list(KERNELS.keys()),
        default=None,
        help="Launch the interactive debugger for one kernel "
        "(Default then CongestionAware sequentially).",
    )
    parser.add_argument(
        "--analyze",
        choices=list(KERNELS.keys()) + ["all"],
        default=None,
        help="Report per-atom move counts (via AtomInterpreter) for one "
        "kernel or 'all' to compare distributions Default vs CongestionAware.",
    )
    args = parser.parse_args()

    if args.visualize is not None:
        kernel, name = KERNELS[args.visualize]
        visualize(kernel, name)
    elif args.analyze is not None:
        targets = (
            list(KERNELS.values()) if args.analyze == "all" else [KERNELS[args.analyze]]
        )
        for kernel, name in targets:
            analyze_on(kernel, name)
    else:
        compare_on(ghz_6, KERNELS["ghz_6"][1])
        compare_on(ghz_optimal, KERNELS["ghz_optimal"][1])
        compare_on(steane_physical_35, KERNELS["steane"][1])


if __name__ == "__main__":
    main()
