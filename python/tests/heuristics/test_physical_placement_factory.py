from __future__ import annotations

import pytest

from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
from bloqade.lanes.bytecode._native import SearchStrategy
from bloqade.lanes.heuristics.physical import make_physical_placement_strategy
from bloqade.lanes.heuristics.physical.nohome import NoHomePlacementStrategy


def test_make_physical_placement_strategy_defaults_match_physical_pipeline():
    strategy = make_physical_placement_strategy()

    assert isinstance(strategy, PalindromePlacementStrategy)
    inner = strategy.inner
    assert isinstance(inner, NoHomePlacementStrategy)
    assert inner.strategy == SearchStrategy.ENTROPY
    assert inner.k_candidates == 3
    assert inner.max_expansions == 300
    assert inner.lambda_lookahead == 0.0


def test_make_physical_placement_strategy_threads_user_knobs():
    strategy = make_physical_placement_strategy(
        move_solutions_per_layer=8,
        search_budget=10_000,
        strategy="astar",
    )

    assert isinstance(strategy, PalindromePlacementStrategy)
    inner = strategy.inner
    assert isinstance(inner, NoHomePlacementStrategy)
    assert inner.strategy == SearchStrategy.ASTAR
    assert inner.k_candidates == 8
    assert inner.max_expansions == 10_000
    assert inner.lambda_lookahead == 0.0


def test_make_physical_placement_strategy_can_disable_return_moves():
    strategy = make_physical_placement_strategy(return_moves=False)

    assert isinstance(strategy, NoHomePlacementStrategy)


def test_make_physical_placement_strategy_allows_unbounded_search_budget():
    strategy = make_physical_placement_strategy(search_budget=None)

    assert isinstance(strategy, PalindromePlacementStrategy)
    inner = strategy.inner
    assert isinstance(inner, NoHomePlacementStrategy)
    assert inner.max_expansions is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"move_solutions_per_layer": 0}, "move_solutions_per_layer"),
        ({"search_budget": 0}, "search_budget"),
    ],
)
def test_make_physical_placement_strategy_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_physical_placement_strategy(**kwargs)
