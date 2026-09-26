"""Strategy definitions and case-by-strategy expansion helpers."""

from __future__ import annotations

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
from bloqade.lanes.bytecode import MoverSelection
from bloqade.lanes.heuristics.physical import (
    NoReturnPlacementStrategy,
    RecedingHorizonNoReturnPlacementStrategy,
    make_physical_placement_strategy,
)
from bloqade.lanes.heuristics.physical.placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)

# Note: the Move Policy DSL strategy (PolicyPlacementStrategy /
# PolicyTraversal) is intentionally NOT in `default_strategy_configs`.
# It is exercised only by the autotune harness via
# `scripts/autotune/measure_dsl_policy.py`, which constructs the
# strategy directly. Keeping it out of the default benchmark matrix
# means `latest_{physical,logical}.csv` baselines are not contaminated
# by whatever transient candidate.star autotune happens to be iterating.


def default_strategy_configs(
    arch_spec: tuple[str, Callable[[], ArchSpec]] | None = None,
    *,
    include_completion_bound: bool = False,
) -> tuple[StrategyConfig, ...]:
    """Return the default strategy matrix for V1 benchmarks.

    `arch_spec` couples the id and factory so callers cannot tag rows with one
    archspec while building from another. When None, defaults to the built-in
    physical archspec. The factory is invoked once per
    `StrategyConfig.build_placement_strategy` call, preserving the lazy
    construction semantics of the original built-in archspec.

    `include_completion_bound` adds the branch-and-bound variant
    (`rust_entropy_5_bounded`). Off by default, and enabled only for the logical
    suite: on the physical suite the bound spends its full expansion budget
    hunting strictly-better goals on the largest kernel (`adder_64`, 93s ->
    370s) which roughly doubles that suite's runtime, and it buys 8 fewer moves
    there rather than nothing — a trade worth making deliberately, not on every
    CI run.

    Physical bound coverage is measured ad hoc via
    `--architecture physical --strategies rust_entropy_5_bounded`: the CLI turns
    this flag on when `--strategies` names a bounded config, so the strategy is
    selectable by name without joining the default physical matrix.
    """
    if arch_spec is None:
        arch_spec_id = BUILTIN_ARCH_SPEC_ID
        factory: Callable[[], ArchSpec] = physical.get_arch_spec
    else:
        arch_spec_id, factory = arch_spec
    return (
        StrategyConfig(
            strategy_id="pipeline_default",
            backend="rust",
            generator_id="rust_solver",
            # Deliberately unpinned: this row tracks whatever
            # `make_physical_placement_strategy` resolves to, which is what
            # `PhysicalPipeline` gives a user who passes no strategy. Pinning
            # the knobs here would reproduce the blind spot one level down --
            # the point is that a change to the placement family,
            # `backwards_search`, `block_spectators`, or the `search_budget` /
            # `move_solutions_per_layer` defaults shows up as a baseline diff.
            # The factory already wraps its result in
            # PalindromePlacementStrategy when return_moves is on, so this must
            # not be wrapped again.
            build_placement_strategy=lambda: make_physical_placement_strategy(
                arch_spec=factory()
            ),
            arch_spec_id=arch_spec_id,
            notes="shipped PhysicalPipeline default; knobs intentionally unpinned",
        ),
        StrategyConfig(
            strategy_id="pipeline_route_all",
            backend="rust",
            generator_id="rust_solver",
            # The shipped default with NoHome's opt-in mover selection: route
            # every candidate mover assignment and keep the cheapest, against
            # the default's Push-and-Rotate ranking (Epic 4).
            build_placement_strategy=lambda: make_physical_placement_strategy(
                arch_spec=factory(), mover_selection=MoverSelection.ROUTE_ALL
            ),
            arch_spec_id=arch_spec_id,
            notes="pipeline_default with mover_selection=ROUTE_ALL",
        ),
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
            strategy_id="rust_push_rotate",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(strategy="push-rotate"),
                )
            ),
            arch_spec_id=arch_spec_id,
            notes="complete rule-based router, not a search; expands no nodes",
        ),
        StrategyConfig(
            strategy_id="rust_cascade_ids",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(strategy="cascade-ids"),
                )
            ),
            arch_spec_id=arch_spec_id,
            notes="IDS, then an A* refinement gated by the completion bound",
        ),
        StrategyConfig(
            strategy_id="rust_cascade_ids_ungated",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(
                        strategy="cascade-ids", cascade_bound=False
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
            notes="cascade-ids with the refinement's completion-bound gate off",
        ),
        StrategyConfig(
            strategy_id="rust_astar_fallback",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PalindromePlacementStrategy(
                inner=PhysicalPlacementStrategy(
                    arch_spec=factory(),
                    traversal=RustPlacementTraversal(
                        strategy="astar", fallback_push_rotate=True
                    ),
                )
            ),
            arch_spec_id=arch_spec_id,
            notes="A* with Push and Rotate as the reliability net on failure",
        ),
        StrategyConfig(
            strategy_id="rust_loose_goal",
            backend="rust",
            generator_id="rust_solver",
            # No palindrome: the no-return family carries each layer's output
            # layout into the next layer instead of moving atoms back home.
            build_placement_strategy=lambda: NoReturnPlacementStrategy(
                arch_spec=factory()
            ),
            arch_spec_id=arch_spec_id,
            notes="loose-goal entangling solver (LooseGoalCzPlacement), defaults",
        ),
        StrategyConfig(
            strategy_id="rust_receding_horizon",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: RecedingHorizonNoReturnPlacementStrategy(
                arch_spec=factory()
            ),
            arch_spec_id=arch_spec_id,
            notes="receding-horizon loose-goal solver (RecedingHorizonCzPlacement), defaults",
        ),
    ) + (
        (
            StrategyConfig(
                strategy_id="rust_entropy_5_bounded",
                backend="rust",
                generator_id="rust_solver",
                build_placement_strategy=lambda: PalindromePlacementStrategy(
                    inner=PhysicalPlacementStrategy(
                        arch_spec=factory(),
                        traversal=RustPlacementTraversal(
                            strategy="entropy",
                            max_goal_candidates=5,
                            max_expansions=2000,
                            completion_bound="weighted_distance",
                        ),
                    )
                ),
                arch_spec_id=arch_spec_id,
                notes="branch-and-bound pruning with the h0 weighted-distance bound",
            ),
        )
        if include_completion_bound
        else ()
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
