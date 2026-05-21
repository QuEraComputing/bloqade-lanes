"""Strategy definitions and case-by-strategy expansion helpers."""

from __future__ import annotations

import os
from collections.abc import Callable

from benchmarks.harness.models import (
    BUILTIN_ARCH_SPEC_ID,
    BenchmarkCase,
    BenchmarkJob,
    StrategyConfig,
)

from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.arch import ArchSpec
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.heuristics.physical.placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)
from bloqade.lanes.heuristics.physical.policy_movement import (
    PolicyPlacementStrategy,
    PolicyTraversal,
)

DSL_POLICY_ENV_VAR = "BLOQADE_DSL_POLICY"
DEFAULT_DSL_POLICY_PATH = "policies/autotune/candidate.star"


def default_strategy_configs(
    arch_spec: tuple[str, Callable[[], ArchSpec]] | None = None,
) -> tuple[StrategyConfig, ...]:
    """Return the default strategy matrix for V1 benchmarks.

    `arch_spec` couples the id and factory so callers cannot tag rows with one
    archspec while building from another. When None, defaults to the built-in
    physical archspec. The factory is invoked once per
    `StrategyConfig.build_placement_strategy` call, preserving the lazy
    construction semantics of the original built-in archspec.
    """
    if arch_spec is None:
        arch_spec_id = BUILTIN_ARCH_SPEC_ID
        factory: Callable[[], ArchSpec] = physical.get_arch_spec
    else:
        arch_spec_id, factory = arch_spec
    return (
        StrategyConfig(
            strategy_id="rust_entropy_1",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(
                        strategy="entropy", max_goal_candidates=1, max_expansions=2000
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_entropy_5",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(
                        strategy="entropy", max_goal_candidates=5, max_expansions=2000
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_entropy_10",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(
                        strategy="entropy", max_goal_candidates=10, max_expansions=2000
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_entropy_20",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(
                        strategy="entropy", max_goal_candidates=20, max_expansions=2000
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_astar",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(strategy="astar"),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_ids",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(strategy="ids"),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_dfs",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(strategy="dfs"),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_bfs",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(strategy="bfs"),
                )
            ),
            arch_spec_id=arch_spec_id,
        ),
        StrategyConfig(
            strategy_id="rust_greedy",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(
                        strategy="greedy",
                        max_movesets_per_group=50,
                        max_expansions=1000,
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
            notes=(
                "first-solution Rust solve (non-optimal); "
                "Rust solver nodes_explored captured from solver output"
            ),
        ),
        StrategyConfig(
            strategy_id="dsl_autotune",
            backend="rust",
            generator_id="dsl",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PolicyPlacementStrategy(
                    arch_spec=factory(),
                    traversal=PolicyTraversal(
                        policy_path=os.environ.get(
                            DSL_POLICY_ENV_VAR, DEFAULT_DSL_POLICY_PATH
                        ),
                        max_expansions=1000,
                        timeout_s=30.0,
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
            notes=(
                f"DSL policy via .star file; path from ${DSL_POLICY_ENV_VAR} "
                f"(default: {DEFAULT_DSL_POLICY_PATH})"
            ),
        ),
    )


def expand_benchmark_jobs(
    cases: tuple[BenchmarkCase, ...],
    strategies: tuple[StrategyConfig, ...],
    strategy_filter: set[str] | None = None,
) -> list[BenchmarkJob]:
    """Expand case and strategy registries into executable benchmark jobs.

    Cases are taken as-is; the CLI pre-filters them via
    `select_benchmark_cases`, which also expands size buckets. Strategy
    filtering happens here because there is no equivalent strategy
    pre-filter upstream.
    """
    jobs: list[BenchmarkJob] = []
    for case in cases:
        for strategy in strategies:
            if (
                strategy_filter is not None
                and strategy.strategy_id not in strategy_filter
            ):
                continue
            jobs.append(BenchmarkJob(case=case, strategy=strategy))

    jobs.sort(key=lambda job: (job.case.case_id, job.strategy.strategy_id))
    return jobs
