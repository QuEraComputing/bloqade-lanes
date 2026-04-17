from __future__ import annotations

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical.movement import (
    DefaultTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
)


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


def test_default_target_generator_matches_current_rule():
    state = _make_state()
    arch_spec = logical.get_arch_spec()
    ctx = TargetContext(
        arch_spec=arch_spec,
        state=state,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )
    candidates = DefaultTargetGenerator().generate(ctx)
    assert len(candidates) == 1
    target = candidates[0]
    # Target qubit stays put
    assert target[1] == state.layout[1]
    # Control qubit moves to the CZ partner of target's current location
    assert target[0] == arch_spec.get_cz_partner(state.layout[1])


def test_default_target_generator_is_target_generator_abc():
    assert isinstance(DefaultTargetGenerator(), TargetGeneratorABC)
