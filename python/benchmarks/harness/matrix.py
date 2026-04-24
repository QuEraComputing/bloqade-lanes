"""Strategy definitions and case-by-strategy expansion helpers."""

from __future__ import annotations

from benchmarks.harness.models import BenchmarkCase, BenchmarkJob, StrategyConfig

from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.heuristics.physical.placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)


def default_strategy_configs() -> tuple[StrategyConfig, ...]:
    """Return the default strategy matrix for V1 benchmarks."""
    return (
        StrategyConfig(
            strategy_id="rust_entropy_1",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(
                    strategy="entropy", max_goal_candidates=1, max_expansions=2000
                ),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_entropy_5",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(
                    strategy="entropy", max_goal_candidates=5, max_expansions=2000
                ),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_entropy_10",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(
                    strategy="entropy", max_goal_candidates=10, max_expansions=2000
                ),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_entropy_20",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(
                    strategy="entropy", max_goal_candidates=20, max_expansions=2000
                ),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_astar",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(strategy="astar"),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_ids",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(strategy="ids"),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_dfs",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(strategy="dfs"),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_bfs",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(strategy="bfs"),
            ),
        ),
        StrategyConfig(
            strategy_id="rust_greedy",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(
                    strategy="greedy",
                    max_movesets_per_group=50,
                    max_expansions=1000,
                ),
            ),
            notes=(
                "first-solution Rust solve (non-optimal); "
                "Rust solver nodes_explored captured from solver output"
            ),
        ),
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
