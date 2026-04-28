from __future__ import annotations

import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical.movement import PhysicalPlacementStrategy
from bloqade.lanes.heuristics.physical.target_generator import (
    DefaultTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
    TargetGeneratorCallable,
    _CallableTargetGenerator,
    _coerce_target_generator,
    _validate_candidate,
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


# ── _validate_candidate helpers and tests ──


def _make_valid_ctx() -> TargetContext:
    state = _make_state()
    return TargetContext(
        arch_spec=logical.get_arch_spec(),
        state=state,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )


def test_validate_candidate_accepts_default():
    ctx = _make_valid_ctx()
    candidate = DefaultTargetGenerator().generate(ctx)[0]
    _validate_candidate(ctx, candidate)  # no raise


def test_validate_candidate_rejects_missing_qid():
    ctx = _make_valid_ctx()
    candidate = DefaultTargetGenerator().generate(ctx)[0]
    del candidate[1]
    with pytest.raises(ValueError, match="missing"):
        _validate_candidate(ctx, candidate)


def test_validate_candidate_rejects_non_cz_pair():
    ctx = _make_valid_ctx()
    # (0,0) and (2,0) are NOT CZ partners on logical.get_arch_spec()
    # — partner((0,0))=(1,0); partner((2,0))=(3,0). Both checks fail.
    non_partner = layout.LocationAddress(2, 0)
    candidate = {0: ctx.state.layout[0], 1: non_partner}
    with pytest.raises(ValueError, match="blockade"):
        _validate_candidate(ctx, candidate)


def test_validate_candidate_accepts_reverse_pair_direction():
    """The 'either direction' OR-check in _validate_candidate must work
    when partner(control)==target (not just partner(target)==control)."""
    ctx = _make_valid_ctx()
    arch = ctx.arch_spec
    # Build a candidate where the control (qid 0) stays put at its
    # current location, and the target (qid 1) moves to control's partner.
    c_loc = ctx.state.layout[0]
    candidate = {0: c_loc, 1: arch.get_cz_partner(c_loc)}
    _validate_candidate(ctx, candidate)  # no raise


def test_validate_candidate_rejects_unknown_location():
    ctx = _make_valid_ctx()
    # LocationAddress with a wildly out-of-range word_id will fail
    # arch_spec.check_location_group
    bogus = layout.LocationAddress(999, 999)
    candidate = {0: bogus, 1: ctx.state.layout[1]}
    with pytest.raises(ValueError, match="invalid locations"):
        _validate_candidate(ctx, candidate)


def test_validate_candidate_rejects_extra_qid():
    ctx = _make_valid_ctx()
    candidate = DefaultTargetGenerator().generate(ctx)[0]
    candidate[99] = ctx.state.layout[0]
    with pytest.raises(ValueError, match="unexpected"):
        _validate_candidate(ctx, candidate)


def test_validate_candidate_rejects_duplicate_locations():
    """Group-level check catches two qids at the same location."""
    ctx = _make_valid_ctx()
    same_loc = ctx.state.layout[0]
    candidate = {0: same_loc, 1: same_loc}
    with pytest.raises(ValueError, match="invalid locations"):
        _validate_candidate(ctx, candidate)


def test_callable_target_generator_wraps_function():
    ctx = _make_valid_ctx()
    expected = DefaultTargetGenerator().generate(ctx)

    def fn(c: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        assert c is ctx
        return expected

    gen = _CallableTargetGenerator(fn)
    assert isinstance(gen, TargetGeneratorABC)
    assert gen.generate(ctx) == expected


def test_coerce_target_generator_passthrough_for_abc():
    abc_gen = DefaultTargetGenerator()
    assert _coerce_target_generator(abc_gen) is abc_gen


def test_coerce_target_generator_wraps_callable():
    def fn(c: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        return []

    gen = _coerce_target_generator(fn)
    assert isinstance(gen, _CallableTargetGenerator)


def test_coerce_target_generator_returns_none_for_none():
    assert _coerce_target_generator(None) is None


def test_strategy_default_target_generator_is_none():
    s = PhysicalPlacementStrategy()
    assert s._resolved_target_generator is None


def test_strategy_accepts_abc_target_generator():
    gen = DefaultTargetGenerator()
    s = PhysicalPlacementStrategy(target_generator=gen)
    assert s._resolved_target_generator is gen


def test_strategy_wraps_callable_target_generator():
    def fn(ctx: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        return []

    s = PhysicalPlacementStrategy(target_generator=fn)
    assert isinstance(s._resolved_target_generator, _CallableTargetGenerator)


# ── _build_candidates tests ──


def _make_strategy_with_generator(
    gen: TargetGeneratorABC | TargetGeneratorCallable | None,
) -> PhysicalPlacementStrategy:
    return PhysicalPlacementStrategy(
        arch_spec=logical.get_arch_spec(),
        target_generator=gen,
    )


def test_build_candidates_none_returns_default_only():
    strategy = _make_strategy_with_generator(None)
    ctx = _make_valid_ctx()
    candidates = strategy._build_candidates(ctx)
    assert candidates == DefaultTargetGenerator().generate(ctx)


def test_build_candidates_empty_plugin_appends_default():
    def fn(c):
        return []

    strategy = _make_strategy_with_generator(fn)
    ctx = _make_valid_ctx()
    candidates = strategy._build_candidates(ctx)
    assert candidates == DefaultTargetGenerator().generate(ctx)


def test_build_candidates_dedups_default():
    """Plugin returning the default verbatim should yield one candidate."""

    def fn(c):
        return DefaultTargetGenerator().generate(c)

    strategy = _make_strategy_with_generator(fn)
    ctx = _make_valid_ctx()
    candidates = strategy._build_candidates(ctx)
    assert len(candidates) == 1


def test_build_candidates_preserves_plugin_order_with_default_last():
    ctx = _make_valid_ctx()
    default = DefaultTargetGenerator().generate(ctx)[0]
    # Construct a second valid candidate by swapping control/target
    alt = dict(default)
    alt[0], alt[1] = alt[1], alt[0]

    def fn(c):
        return [alt]

    strategy = _make_strategy_with_generator(fn)
    candidates = strategy._build_candidates(ctx)
    assert candidates == [alt, default]


def test_build_candidates_raises_on_malformed():
    def fn(c):
        # (0,0) and (2,0) are NOT CZ partners — partner((0,0))=(1,0), partner((2,0))=(3,0)
        return [{0: c.state.layout[0], 1: layout.LocationAddress(2, 0)}]

    strategy = _make_strategy_with_generator(fn)
    ctx = _make_valid_ctx()
    with pytest.raises(ValueError, match="blockade"):
        strategy._build_candidates(ctx)
