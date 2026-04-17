from __future__ import annotations

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical.movement import TargetContext


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(
            layout.LocationAddress(0, 0),
            layout.LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )


def test_target_context_placement_derives_from_state_layout():
    state = _make_state()
    ctx = TargetContext(
        arch_spec=logical.get_arch_spec(),
        state=state,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )
    assert ctx.placement == {0: state.layout[0], 1: state.layout[1]}
