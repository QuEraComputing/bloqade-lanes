import inspect
from typing import cast

from kirin import ir

from bloqade.gemini import compile as gemini_compile
from bloqade.lanes import compile as compile_api, logical_mvp
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.heuristics.logical_placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import PhysicalGreedyPlacementStrategy


def test_logical_mvp_compile_to_move_uses_logical_defaults(monkeypatch):
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

    monkeypatch.setattr(logical_mvp, "squin_to_move", fake_squin_to_move)

    marker = cast(ir.Method, object())
    out = gemini_compile.compile_squin_to_move(marker)

    assert out == "move_ir"
    assert captured["mt"] is marker
    assert isinstance(
        captured["layout_heuristic"], logical_layout.LogicalLayoutHeuristic
    )
    assert isinstance(captured["placement_strategy"], LogicalPlacementStrategyNoHome)
    assert captured["insert_return_moves"] is True


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
    assert isinstance(captured["placement_strategy"], PhysicalGreedyPlacementStrategy)


def test_physical_compile_has_no_transversal_or_placement_mode():
    params = inspect.signature(compile_api.compile_squin_to_move).parameters
    assert "transversal_rewrite" not in params
    assert "placement_mode" not in params
