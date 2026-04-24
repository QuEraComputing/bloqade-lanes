import inspect
from typing import cast

import pytest
from kirin import ir

from bloqade.lanes import compile as compile_api, logical_mvp
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.heuristics.logical import layout as logical_layout
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.placement import PhysicalPlacementStrategy


def test_logical_mvp_compile_to_move_uses_logical_defaults(monkeypatch):
    captured = {}

    def fake_squin_to_move(
        mt,
        layout_heuristic=None,
        placement_strategy=None,
        insert_return_moves=True,
        no_raise=True,
        logical_initialize=True,
    ):
        captured["mt"] = mt
        captured["layout_heuristic"] = layout_heuristic
        captured["placement_strategy"] = placement_strategy
        captured["insert_return_moves"] = insert_return_moves
        captured["no_raise"] = no_raise
        captured["logical_initialize"] = logical_initialize
        return "move_ir"

    monkeypatch.setattr(logical_mvp, "squin_to_move", fake_squin_to_move)

    marker = cast(ir.Method, object())
    out = logical_mvp.compile_squin_to_move(marker)

    assert out == "move_ir"
    assert captured["mt"] is marker
    assert isinstance(
        captured["layout_heuristic"], logical_layout.LogicalLayoutHeuristic
    )
    assert isinstance(captured["placement_strategy"], LogicalPlacementStrategyNoHome)
    assert captured["insert_return_moves"] is True
    assert captured["logical_initialize"] is True


def test_modular_compile_to_move_allows_strategy_swapping(monkeypatch):
    captured = {}

    def fake_squin_to_move(
        mt,
        layout_heuristic=None,
        placement_strategy=None,
        insert_return_moves=True,
        no_raise=True,
    ):
        captured["mt"] = mt
        captured["layout_heuristic"] = layout_heuristic
        captured["placement_strategy"] = placement_strategy
        captured["insert_return_moves"] = insert_return_moves
        captured["no_raise"] = no_raise
        return "move_ir"

    monkeypatch.setattr(compile_api, "squin_to_move", fake_squin_to_move)

    marker = cast(ir.Method, object())
    custom_layout = cast(LayoutHeuristicABC, object())
    custom_strategy = cast(PlacementStrategyABC, object())
    out = compile_api.compile_squin_to_move(
        marker,
        layout_heuristic=custom_layout,
        placement_strategy=custom_strategy,
        insert_return_moves=False,
    )

    assert out == "move_ir"
    assert captured["mt"] is marker
    assert captured["layout_heuristic"] is custom_layout
    assert captured["placement_strategy"] is custom_strategy
    assert captured["insert_return_moves"] is False


def test_physical_compile_to_move_defaults_are_physical(monkeypatch):
    captured = {}

    def fake_squin_to_move(
        mt,
        layout_heuristic=None,
        placement_strategy=None,
        insert_return_moves=True,
        no_raise=True,
    ):
        captured["layout_heuristic"] = layout_heuristic
        captured["placement_strategy"] = placement_strategy
        return "move_ir"

    monkeypatch.setattr(compile_api, "squin_to_move", fake_squin_to_move)

    marker = cast(ir.Method, object())
    out = compile_api.compile_squin_to_move(marker)
    assert out == "move_ir"
    assert isinstance(
        captured["layout_heuristic"], PhysicalLayoutHeuristicGraphPartitionCenterOut
    )
    assert isinstance(captured["placement_strategy"], PhysicalPlacementStrategy)


def test_physical_compile_has_no_transversal_or_placement_mode():
    params = inspect.signature(compile_api.compile_squin_to_move).parameters
    assert "transversal_rewrite" not in params
    assert "placement_mode" not in params


# --- compile_squin_to_move_best ---------------------------------------------


def _race_stubs(monkeypatch, events_by_id: dict[int, int]) -> list[tuple]:
    """Wire fakes so `compile_squin_to_move` returns a sentinel tagged with
    the strategy identity, and `_count_move_events` reads a pre-computed
    event count looked up by ``id(strategy)``. Returns the per-call log.
    """
    calls: list[tuple] = []

    class _Sentinel:
        def __init__(self, strategy):
            self.strategy = strategy
            self.events = events_by_id[id(strategy)]

    def fake_compile(mt, *, placement_strategy=None, **_kw):
        calls.append((mt, placement_strategy))
        return _Sentinel(placement_strategy)

    def fake_count(mt):
        return mt.events

    monkeypatch.setattr(compile_api, "compile_squin_to_move", fake_compile)
    monkeypatch.setattr(compile_api, "_count_move_events", fake_count)
    return calls


def test_compile_squin_to_move_best_picks_fewest_events(monkeypatch):
    strat_a = cast(PlacementStrategyABC, object())
    strat_b = cast(PlacementStrategyABC, object())
    strat_c = cast(PlacementStrategyABC, object())
    _race_stubs(monkeypatch, {id(strat_a): 10, id(strat_b): 5, id(strat_c): 8})

    marker = cast(ir.Method, object())
    out_mt, label = compile_api.compile_squin_to_move_best(
        marker,
        strategies=[("A", strat_a), ("B", strat_b), ("C", strat_c)],
    )

    assert label == "B"
    assert out_mt.strategy is strat_b  # type: ignore[attr-defined]
    assert out_mt.events == 5  # type: ignore[attr-defined]


def test_compile_squin_to_move_best_tie_break_earliest_in_list(monkeypatch):
    strat_a = cast(PlacementStrategyABC, object())
    strat_b = cast(PlacementStrategyABC, object())
    strat_c = cast(PlacementStrategyABC, object())
    # All three tie at 7; earliest wins.
    _race_stubs(monkeypatch, {id(strat_a): 7, id(strat_b): 7, id(strat_c): 7})

    marker = cast(ir.Method, object())
    _, label = compile_api.compile_squin_to_move_best(
        marker,
        strategies=[("A", strat_a), ("B", strat_b), ("C", strat_c)],
    )
    assert label == "A"


def test_compile_squin_to_move_best_runs_every_strategy(monkeypatch):
    strat_a = cast(PlacementStrategyABC, object())
    strat_b = cast(PlacementStrategyABC, object())
    calls = _race_stubs(monkeypatch, {id(strat_a): 3, id(strat_b): 4})

    marker = cast(ir.Method, object())
    compile_api.compile_squin_to_move_best(
        marker,
        strategies=[("A", strat_a), ("B", strat_b)],
    )

    assert [c[1] for c in calls] == [strat_a, strat_b]


def test_compile_squin_to_move_best_empty_strategies_raises():
    marker = cast(ir.Method, object())
    with pytest.raises(ValueError, match="at least one strategy"):
        compile_api.compile_squin_to_move_best(marker, strategies=[])
