from dataclasses import dataclass, field
from typing import cast

from kirin import ir
from kirin.analysis.forward import ForwardFrame

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    PlacementStrategyABC,
)
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import place
from bloqade.lanes.layout import ArchSpec, LocationAddress


@dataclass
class CaptureLookaheadStrategy(PlacementStrategyABC):
    arch_spec: ArchSpec = field(default_factory=get_arch_spec)
    captured_lookahead: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]] = (
        field(default_factory=list)
    )

    def validate_initial_layout(
        self, initial_layout: tuple[LocationAddress, ...]
    ) -> None:
        _ = initial_layout

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        _ = controls, targets
        self.captured_lookahead.append(lookahead_cz_layers)
        return state

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        _ = qubits
        return state

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        _ = qubits
        return state


class _FakeFrame:
    def __init__(self, state: ConcreteState):
        self._state = state

    def get(self, _value):
        return self._state


def test_impl_cz_forwards_buffered_lookahead_layers():
    """Test that impl_cz forwards the buffered lookahead layers to the placement strategy."""
    strategy = CaptureLookaheadStrategy()
    analysis = object.__new__(place.PlacementAnalysis)
    analysis.placement_strategy = strategy
    analysis.cz_lookahead_buffers = {}
    analysis.cz_lookahead_stmt_positions = {}
    block = ir.Block(
        [
            cz0 := place.CZ(ir.TestValue(), qubits=(0, 1)),
            cz1 := place.CZ(cz0.state_after, qubits=(2, 3)),
        ]
    )
    analysis.cz_lookahead_buffers[block] = analysis.build_cz_buffer(block)

    frame = _FakeFrame(
        ConcreteState(
            occupied=frozenset(),
            layout=(
                LocationAddress(0, 0, 0),
                LocationAddress(0, 0, 1),
                LocationAddress(0, 0, 2),
                LocationAddress(0, 0, 3),
            ),
            move_count=(0, 0, 0, 0),
        )
    )
    methods = place.PlacementMethods()

    methods.impl_cz(analysis, cast(ForwardFrame[AtomState], frame), cz0)
    methods.impl_cz(analysis, cast(ForwardFrame[AtomState], frame), cz1)

    assert strategy.captured_lookahead == [
        (((0,), (1,)), ((2,), (3,))),
        (((2,), (3,)),),
    ]
