"""Evaluation script: compare baseline vs loose-goal placement on random CZ circuits.

Uses the Gemini physical architecture (20 words, 8 sites/word).
Compares PhysicalPlacementStrategy (Python entropy traversal) against
LooseGoalPlacementStrategy (Rust IDS with multi-restart).

Usage:
    python scripts/eval_pipeline.py [--n-qubits 4] [--depth 3] [--seeds 3]
    python scripts/eval_pipeline.py --n-qubits 4 --depth 3 --seeds 1 --visualize
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import random
import sys
import tempfile
import time

from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.compile import squin_to_move
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.loose_goal import LooseGoalPlacementStrategy
from bloqade.lanes.heuristics.physical.movement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)
from bloqade.lanes.upstream import always_merge_heuristic

# ── Random circuit generation ──────────────────────────────────────


def make_random_cz_pairs(
    n_qubits: int, depth: int, seed: int, max_pairs: int | None = None
) -> list[list[tuple[int, int]]]:
    """Generate random CZ pair layers.

    If ``max_pairs`` is set, each layer contains at most that many
    simultaneous CZ pairs.
    """
    rng = random.Random(seed)
    layers = []
    for _ in range(depth):
        indices = list(range(n_qubits))
        rng.shuffle(indices)
        pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        layers.append(pairs)
    return layers


def make_kernel(n_qubits: int, cz_layers: list[list[tuple[int, int]]]):
    """Create a squin kernel with CZ gates via temp file.

    The Kirin IR framework requires ``inspect.getsource`` on kernel
    functions, so we write to a temp file and import it.
    """
    lines = [
        "from kirin.dialects import ilist",
        "from bloqade import squin",
        "",
        "@squin.kernel",
        "def random_circuit():",
        f"    q = squin.qalloc({n_qubits})",
    ]

    for layer in cz_layers:
        if len(layer) == 1:
            c, t = layer[0]
            lines.append(f"    squin.cz(q[{c}], q[{t}])")
        elif len(layer) > 1:
            controls = ", ".join(f"q[{c}]" for c, _ in layer)
            targets = ", ".join(f"q[{t}]" for _, t in layer)
            lines.append(
                f"    squin.broadcast.cz(ilist.IList([{controls}]), ilist.IList([{targets}]))"
            )

    lines.append("")

    code = "\n".join(lines)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="eval_kernel_", delete=False
    ) as f:
        f.write(code)
        f.flush()
        spec = importlib.util.spec_from_file_location("_eval_kernel", f.name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_eval_kernel"] = mod
        spec.loader.exec_module(mod)

    return mod.random_circuit


# ── Pipeline runner ────────────────────────────────────────────────


def count_move_layers(move_ir) -> int:
    """Count Move statements in the compiled IR."""
    count = 0
    for stmt in move_ir.callable_region.walk():
        if isinstance(stmt, move.Move):
            count += 1
    return count


def count_total_lanes(move_ir) -> int:
    """Count total lane operations across all Move statements."""
    total = 0
    for stmt in move_ir.callable_region.walk():
        if isinstance(stmt, move.Move):
            total += len(stmt.lanes)
    return total


def run_pipeline(
    n_qubits: int,
    depth: int,
    seed: int,
    placement_strategy,
    layout_heuristic,
    insert_return_moves: bool = True,
    max_pairs: int | None = None,
    merge_heuristic=None,
):
    """Run the full compilation pipeline on a random circuit."""
    cz_layers = make_random_cz_pairs(n_qubits, depth, seed, max_pairs=max_pairs)
    kernel = make_kernel(n_qubits, cz_layers)

    n_cz_pairs = sum(len(layer) for layer in cz_layers)

    kwargs = {}
    if merge_heuristic is not None:
        kwargs["merge_heuristic"] = merge_heuristic

    start = time.time()
    move_ir = squin_to_move(
        kernel,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_return_moves=insert_return_moves,
        logical_initialize=False,
        **kwargs,
    )
    elapsed_ms = (time.time() - start) * 1000

    n_moves = count_move_layers(move_ir)
    n_lanes = count_total_lanes(move_ir)

    return {
        "seed": seed,
        "n_qubits": n_qubits,
        "depth": depth,
        "n_cz_pairs": n_cz_pairs,
        "n_move_layers": n_moves,
        "n_total_lanes": n_lanes,
        "time_ms": elapsed_ms,
        "move_ir": move_ir,
    }


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs loose-goal placement on random CZ circuits"
    )
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=3, help="Number of CZ layers")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=300,
        help="Search budget per CZ layer (baseline)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Max simultaneous CZ pairs per layer (default: unlimited)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the first seed's result (loose-goal)",
    )
    args = parser.parse_args()

    arch_spec = get_physical_arch_spec()
    print(
        f"Architecture: {len(arch_spec.words)} words, {arch_spec.sites_per_word} sites/word"
    )
    print(f"Circuit: {args.n_qubits} qubits, depth {args.depth}, {args.seeds} seeds")
    print()

    layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=arch_spec
    )

    strategies = {
        "Baseline (Rust IDS r=20)": {
            "strategy": PhysicalPlacementStrategy(
                arch_spec=arch_spec,
                traversal=RustPlacementTraversal(
                    strategy="ids",
                    max_expansions=args.max_expansions,
                    restarts=20,
                    lookahead=True,
                ),
            ),
            "insert_return_moves": True,
        },
        "Loose-Goal (IDS r=20)": {
            "strategy": LooseGoalPlacementStrategy(
                arch_spec=arch_spec,
                strategy="ids",
                max_expansions=args.max_expansions,
                restarts=20,
                dynamic_targets=True,
                recompute_interval=0,
            ),
            "insert_return_moves": False,
            "merge_heuristic": always_merge_heuristic,
        },
    }

    all_strategy_results: dict[str, list[dict]] = {}

    for strat_name, strat_config in strategies.items():
        print(f"=== {strat_name} ===\n")

        header = f"{'Seed':>4} {'Qubits':>6} {'CZ pairs':>8} {'Moves':>6} {'Lanes':>6} {'Time(ms)':>10}"
        print(header)
        print("─" * len(header))

        results = []
        for seed in range(1, args.seeds + 1):
            try:
                result = run_pipeline(
                    args.n_qubits,
                    args.depth,
                    seed,
                    strat_config["strategy"],
                    layout_heuristic,
                    insert_return_moves=strat_config["insert_return_moves"],
                    max_pairs=args.max_pairs,
                    merge_heuristic=strat_config.get("merge_heuristic"),
                )
                failed = result["n_move_layers"] == 0 and args.depth > 0
                results.append(result)
                if failed:
                    print(
                        f"{seed:>4} {result['n_qubits']:>6} {result['n_cz_pairs']:>8} "
                        f"{'FAIL':>6} {'':>6} "
                        f"{result['time_ms']:>10.1f}"
                    )
                else:
                    print(
                        f"{seed:>4} {result['n_qubits']:>6} {result['n_cz_pairs']:>8} "
                        f"{result['n_move_layers']:>6} {result['n_total_lanes']:>6} "
                        f"{result['time_ms']:>10.1f}"
                    )
            except Exception as e:
                print(f"{seed:>4} {args.n_qubits:>6}   ERROR: {e}")
                import traceback

                traceback.print_exc()

        all_strategy_results[strat_name] = results

        if results:
            solved = [r for r in results if r["n_move_layers"] > 0 or args.depth == 0]
            n_fail = len(results) - len(solved)
            if solved:
                avg_moves = sum(r["n_move_layers"] for r in solved) / len(solved)
                avg_lanes = sum(r["n_total_lanes"] for r in solved) / len(solved)
                avg_time = sum(r["time_ms"] for r in results) / len(results)
                fail_str = f" ({n_fail} failed)" if n_fail else ""
                print(
                    f"\nAvg: {avg_moves:.1f} move layers, "
                    f"{avg_lanes:.1f} lanes, {avg_time:.1f} ms"
                    f" [{len(solved)}/{len(results)} solved{fail_str}]"
                )
            else:
                print(f"\nAll {len(results)} seeds failed.")

        print()

    # Summary comparison.
    print("=" * 70)
    print("Summary:")
    print(
        f"{'Strategy':<25} {'Avg Moves':>10} {'Avg Lanes':>10} {'Avg Time':>10} {'Solved':>8}"
    )
    print("─" * 70)
    for strat_name, results in all_strategy_results.items():
        if not results:
            print(f"{strat_name:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
            continue
        solved = [r for r in results if r["n_move_layers"] > 0 or args.depth == 0]
        if solved:
            avg_moves = sum(r["n_move_layers"] for r in solved) / len(solved)
            avg_lanes = sum(r["n_total_lanes"] for r in solved) / len(solved)
        else:
            avg_moves = avg_lanes = 0.0
        avg_time = sum(r["time_ms"] for r in results) / len(results)
        solved_str = f"{len(solved)}/{len(results)}"
        print(
            f"{strat_name:<25} {avg_moves:>10.1f} {avg_lanes:>10.1f}"
            f" {avg_time:>8.0f} ms {solved_str:>8}"
        )

    # Visualize best run from each strategy.
    if args.visualize:
        from bloqade.lanes import visualize

        for strat_name, results in all_strategy_results.items():
            if not results:
                continue
            best = min(results, key=lambda r: r["n_move_layers"])
            print(
                f"\nVisualizing best {strat_name} "
                f"(seed {best['seed']}, {best['n_move_layers']} moves)..."
            )
            visualize.debugger(
                best["move_ir"], arch_spec, interactive=True, atom_marker="o"
            )


if __name__ == "__main__":
    main()
