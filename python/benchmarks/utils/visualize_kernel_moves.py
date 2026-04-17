"""Visualize compiled move programs for benchmark kernels."""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Literal, cast

from kirin import ir

from bloqade.lanes import visualize
from bloqade.lanes.arch.gemini import logical as logical_arch, physical as physical_arch
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)
from bloqade.lanes.upstream import squin_to_move

ArchMode = Literal["logical", "physical"]
RustStrategy = Literal[
    "astar",
    "dfs",
    "bfs",
    "greedy",
    "ids",
    "cascade",
    "cascade-ids",
    "cascade-dfs",
    "cascade-entropy",
    "entropy",
]


def _load_case_kernel(*, bucket: str, case_id: str) -> ir.Method:
    module_name = f"benchmarks.kernels.{bucket}.{case_id}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Cannot import case module '{module_name}'. "
            "Check --bucket and --case-id values."
        ) from exc

    kernels = [value for value in vars(module).values() if isinstance(value, ir.Method)]
    if len(kernels) != 1:
        raise ValueError(
            f"Expected exactly one kernel in '{module_name}', found {len(kernels)}."
        )
    return kernels[0]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize move programs generated for a benchmark kernel."
    )
    parser.add_argument(
        "--case-id",
        required=True,
        help="Kernel case ID (example: multi_qubit_rb_10_9_Z_XXZXYXYZ).",
    )
    parser.add_argument(
        "--bucket",
        default="random_stabilizers",
        help=(
            "Kernel bucket under benchmarks.kernels (default: random_stabilizers). "
            "Examples: small, medium, large, random_stabilizers."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("logical", "physical"),
        default="logical",
        help="Compile in logical or physical mode (default: logical).",
    )
    parser.add_argument(
        "--strategy",
        choices=(
            "astar",
            "dfs",
            "bfs",
            "greedy",
            "ids",
            "cascade",
            "cascade-ids",
            "cascade-dfs",
            "cascade-entropy",
            "entropy",
        ),
        default="astar",
        help="Rust search strategy (default: astar).",
    )
    parser.add_argument(
        "--animated",
        action="store_true",
        help="Use animated visualizer instead of step debugger.",
    )
    return parser


def _compile_move_kernel(
    *,
    case_kernel: ir.Method,
    architecture: ArchMode,
    strategy: RustStrategy,
):
    if architecture == "logical":
        arch_spec = logical_arch.get_arch_spec()
        layout_heuristic = logical_layout.LogicalLayoutHeuristic()
        logical_initialize = True
    else:
        arch_spec = physical_arch.get_arch_spec()
        layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(
            arch_spec=arch_spec
        )
        logical_initialize = False

    placement_strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        traversal=RustPlacementTraversal(strategy=strategy),
    )
    move_mt = squin_to_move(
        case_kernel,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        logical_initialize=logical_initialize,
    )
    return move_mt, arch_spec


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        case_kernel = _load_case_kernel(bucket=args.bucket, case_id=args.case_id)
    except ValueError as exc:
        parser.error(str(exc))

    move_mt, arch_spec = _compile_move_kernel(
        case_kernel=case_kernel,
        architecture=cast(ArchMode, args.architecture),
        strategy=cast(RustStrategy, args.strategy),
    )
    if args.animated:
        visualize.animated_debugger(move_mt, arch_spec, interactive=True)
    else:
        visualize.debugger(move_mt, arch_spec, interactive=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
