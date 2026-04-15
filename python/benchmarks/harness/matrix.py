"""Strategy definitions and case-by-strategy expansion helpers."""

from __future__ import annotations

from benchmarks.harness.models import BenchmarkCase, BenchmarkJob, StrategyConfig

from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.heuristics.physical_placement import (
    EntropyPlacementTraversal,
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
    SearchStrategy,
)


def default_strategy_configs() -> tuple[StrategyConfig, ...]:
    """Return the default strategy matrix for V1 benchmarks."""
    return (
        StrategyConfig(
            strategy_id="python_entropy",
            backend="python",
            generator_id="heuristic",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=EntropyPlacementTraversal(),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_astar",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(strategy=SearchStrategy.ASTAR),
            ),
        ),
        # StrategyConfig(
        #     strategy_id="rust_greedy_unbounded",
        #     backend="rust",
        #     generator_id="rust_solver",
        #     build_placement_strategy=lambda: PhysicalPlacementStrategy(
        #         arch_spec=physical.get_arch_spec(),
        #         traversal=RustPlacementTraversal(
        #             strategy=SearchStrategy.GREEDY,
        #             top_c=50,
        #             max_movesets_per_group=50,
        #             max_expansions=300,
        #         ),
        #     ),
        #     notes=(
        #         "first-solution Rust solve (non-optimal); "
        #         "Rust solver nodes_explored captured from solver output"
        #     ),
        # ),
        # StrategyConfig(
        #     strategy_id="python_bfs",
        #     backend="python",
        #     generator_id="exhaustive",
        #     build_placement_strategy=lambda: PhysicalPlacementStrategy(
        #         arch_spec=physical.get_arch_spec(),
        #         traversal=BFSPlacementTraversal(),
        #     ),
        # ),
        # StrategyConfig(
        #     strategy_id="python_greedy_best_first",
        #     backend="python",
        #     generator_id="exhaustive",
        #     build_placement_strategy=lambda: PhysicalPlacementStrategy(
        #         arch_spec=physical.get_arch_spec(),
        #         traversal=GreedyPlacementTraversal(),
        #     ),
        # ),
    )


def expand_benchmark_jobs(
    cases: tuple[BenchmarkCase, ...],
    strategies: tuple[StrategyConfig, ...],
    case_filter: set[str] | None = None,
    strategy_filter: set[str] | None = None,
) -> list[BenchmarkJob]:
    """Expand case and strategy registries into executable benchmark jobs."""
    jobs: list[BenchmarkJob] = []
    for case in cases:
        if case_filter is not None and case.case_id not in case_filter:
            continue
        for strategy in strategies:
            if (
                strategy_filter is not None
                and strategy.strategy_id not in strategy_filter
            ):
                continue
            jobs.append(BenchmarkJob(case=case, strategy=strategy))

    jobs.sort(key=lambda job: (job.case.case_id, job.strategy.strategy_id))
    return jobs
