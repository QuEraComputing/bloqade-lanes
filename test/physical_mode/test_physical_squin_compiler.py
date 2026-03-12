import math
from dataclasses import dataclass

import pytest
from bloqade.gemini import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.logical_layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical_placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import PhysicalGreedyPlacementStrategy
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.logical_mvp import compile_squin_to_move


def _count_move_events(mt) -> int:
    return sum(1 for stmt in mt.callable_region.walk() if isinstance(stmt, move.Move))


@gemini_logical.kernel(aggressive_unroll=True)
def _physical_compile_kernel():
    reg = qubit.qalloc(5)
    squin.broadcast.u3(0.25 * math.pi, 0.1 * math.pi, 0.0, reg)
    squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))


def test_compile_squin_to_move_physical_mode_smoke():
    physical_move = compile_squin_to_move(
        _physical_compile_kernel,
        no_raise=False,
        placement_mode="physical",
    )
    assert _count_move_events(physical_move) >= 0


def test_compile_squin_to_move_physical_mode_uses_physical_strategies():
    @dataclass
    class TrackingLayoutHeuristic(PhysicalLayoutHeuristicGraphPartitionCenterOut):
        compute_calls: int = 0

        def compute_layout(self, all_qubits, stages):
            self.compute_calls += 1
            return super().compute_layout(all_qubits, stages)

    @dataclass
    class TrackingMovementStrategy(PhysicalGreedyPlacementStrategy):
        cz_calls: int = 0

        def cz_placements(
            self,
            state,
            controls,
            targets,
            lookahead_cz_layers=(),
        ):
            self.cz_calls += 1
            return super().cz_placements(
                state,
                controls,
                targets,
                lookahead_cz_layers=lookahead_cz_layers,
            )

    arch_spec = generate_arch_hypercube(4)
    layout_heuristic = TrackingLayoutHeuristic(arch_spec=arch_spec)
    movement_strategy = TrackingMovementStrategy(arch_spec=arch_spec)
    physical_move = compile_squin_to_move(
        _physical_compile_kernel,
        no_raise=False,
        placement_mode="physical",
        layout_heuristic=layout_heuristic,
        placement_strategy=movement_strategy,
    )
    assert _count_move_events(physical_move) >= 0
    assert layout_heuristic.compute_calls > 0
    assert movement_strategy.cz_calls > 0


def test_compile_squin_to_move_physical_mode_respects_explicit_overrides():
    @dataclass
    class TrackingLayoutHeuristic(LayoutHeuristicABC):
        called: bool = False

        def compute_layout(self, all_qubits, stages):
            self.called = True
            _ = stages
            qubits = tuple(sorted(all_qubits))
            return tuple(LocationAddress(0, idx) for idx, _ in enumerate(qubits))

    @dataclass
    class TrackingMovementStrategy(PhysicalGreedyPlacementStrategy):
        cz_calls: int = 0

        def cz_placements(
            self,
            state,
            controls,
            targets,
            lookahead_cz_layers=(),
        ):
            self.cz_calls += 1
            return super().cz_placements(
                state,
                controls,
                targets,
                lookahead_cz_layers=lookahead_cz_layers,
            )

    arch_spec = generate_arch_hypercube(4)
    layout_heuristic = TrackingLayoutHeuristic()
    movement_strategy = TrackingMovementStrategy(arch_spec=arch_spec)
    physical_move = compile_squin_to_move(
        _physical_compile_kernel,
        no_raise=False,
        placement_mode="physical",
        layout_heuristic=layout_heuristic,
        placement_strategy=movement_strategy,
    )
    assert _count_move_events(physical_move) >= 0
    assert layout_heuristic.called
    assert movement_strategy.cz_calls > 0


def test_compile_squin_to_move_rejects_layout_mode_mismatch():
    with pytest.raises(ValueError, match="layout_heuristic is incompatible"):
        compile_squin_to_move(
            _physical_compile_kernel,
            no_raise=False,
            placement_mode="physical",
            layout_heuristic=LogicalLayoutHeuristic(),
            placement_strategy=PhysicalGreedyPlacementStrategy(
                arch_spec=generate_arch_hypercube(4)
            ),
        )


def test_compile_squin_to_move_rejects_strategy_mode_mismatch():
    with pytest.raises(ValueError, match="placement_strategy is incompatible"):
        compile_squin_to_move(
            _physical_compile_kernel,
            no_raise=False,
            placement_mode="physical",
            placement_strategy=LogicalPlacementStrategyNoHome(),
        )
