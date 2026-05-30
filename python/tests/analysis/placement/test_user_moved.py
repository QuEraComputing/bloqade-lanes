"""Tests for UserMoved lattice element and ExecuteCZReturn user_move_layers extension."""

from bloqade.lanes.analysis.placement.lattice import (
    AtomState,
    ConcreteState,
    ExecuteCZReturn,
    UserMoved,
)
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    ZoneAddress,
)


def _loc(z: int, w: int, s: int) -> LocationAddress:
    return LocationAddress(zone_id=z, word_id=w, site_id=s)


def _lane(z: int, w: int, s: int, d: Direction = Direction.FORWARD) -> SiteLaneAddress:
    return SiteLaneAddress(z, w, s, d)


def _concrete(layout):
    return ConcreteState(
        occupied=frozenset(),
        layout=layout,
        move_count=(0,) * len(layout),
    )


# --- Strategy tests using the logical arch ---

# The logical arch uses home positions at zone_id=0, even word_ids (0,2,4,...), site_id=0.
# We use those as valid addresses throughout.


def _make_strategy():
    from bloqade.lanes.arch.gemini import logical as logical_arch
    from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategy

    return LogicalPlacementStrategy(arch_spec=logical_arch.get_arch_spec())


def test_move_to_placements_returns_user_moved():
    strat = _make_strategy()
    layout = (
        _loc(0, 0, 0),
        _loc(0, 2, 0),
    )
    state = _concrete(layout)
    dest = _loc(0, 4, 0)  # empty home slot
    result = strat.move_to_placements(state, qubits=(0,), locations=(dest,))
    assert isinstance(result, UserMoved)
    assert result.layout[0] == dest
    assert result.pre_user_layout == layout
    assert result.accumulated_move_layers == result.move_layers


def test_move_to_placements_occupancy_conflict_returns_bottom():
    strat = _make_strategy()
    layout = (
        _loc(0, 0, 0),
        _loc(0, 2, 0),
    )
    state = _concrete(layout)
    # Try to move qubit 0 into qubit 1's current slot — conflict
    dest = _loc(0, 2, 0)
    result = strat.move_to_placements(state, qubits=(0,), locations=(dest,))
    assert result == AtomState.bottom()


def test_sq_placements_user_moved_preserves_state():
    """SQ gates must not corrupt UserMoved; the state passes through unchanged."""
    strat = _make_strategy()
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    um = UserMoved.from_concrete_state(
        _concrete(layout),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=layout,
    )
    result = strat.sq_placements(um, qubits=(0,))
    assert result is um  # UserMoved passes through SQ gates unchanged


def test_measure_placements_user_moved_returns_bottom():
    strat = _make_strategy()
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    um = UserMoved.from_concrete_state(
        _concrete(layout),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=layout,
    )
    result = strat.measure_placements(um, qubits=(0, 1))
    assert result == AtomState.bottom()


def test_consecutive_move_to_accumulates():
    strat = _make_strategy()
    layout = (
        _loc(0, 0, 0),
        _loc(0, 2, 0),
        _loc(0, 4, 0),
    )
    state = _concrete(layout)
    dest1 = _loc(0, 6, 0)
    um1 = strat.move_to_placements(state, qubits=(0,), locations=(dest1,))
    assert isinstance(um1, UserMoved)
    assert um1.pre_user_layout == layout

    dest2 = _loc(0, 8, 0)
    um2 = strat.move_to_placements(um1, qubits=(1,), locations=(dest2,))
    assert isinstance(um2, UserMoved)
    assert um2.pre_user_layout == layout  # unchanged: the original home
    assert len(um2.accumulated_move_layers) >= len(um1.accumulated_move_layers)


def test_user_moved_is_concrete_state():
    um = UserMoved.from_concrete_state(
        _concrete((_loc(0, 0, 0), _loc(0, 0, 1))),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=(_loc(0, 0, 0), _loc(0, 0, 1)),
    )
    assert isinstance(um, ConcreteState)


def test_user_moved_get_move_layers_returns_own_layers():
    layers = ((_lane(0, 0, 0),),)
    um = UserMoved.from_concrete_state(
        _concrete((_loc(0, 1, 0), _loc(0, 0, 1))),
        move_layers=layers,
        accumulated_move_layers=layers,
        pre_user_layout=(_loc(0, 0, 0), _loc(0, 0, 1)),
    )
    assert um.get_move_layers() == layers


def test_user_moved_get_reverse_moves_empty():
    um = UserMoved.from_concrete_state(
        _concrete((_loc(0, 1, 0),)),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=(_loc(0, 0, 0),),
    )
    assert um.get_reverse_moves() == ()


def test_execute_cz_return_user_move_layers_default_empty():
    ecz = ExecuteCZReturn(
        occupied=frozenset(),
        layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
        move_count=(0, 0),
        active_cz_zones=frozenset([ZoneAddress(0)]),
        move_layers=((_lane(0, 0, 0),),),
        initial_layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
    )
    assert ecz.user_move_layers == ()
    assert len(ecz.return_move_layers) == 1  # only compiler reverse


def test_execute_cz_return_with_user_move_layers():
    compiler_lane = _lane(0, 0, 0)
    user_lane = _lane(0, 0, 1)
    ecz = ExecuteCZReturn(
        occupied=frozenset(),
        layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
        move_count=(0, 0),
        active_cz_zones=frozenset([ZoneAddress(0)]),
        move_layers=((compiler_lane,),),
        user_move_layers=((user_lane,),),
        initial_layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
    )
    assert len(ecz.return_move_layers) == 2  # compiler_reverse + user_reverse
    assert ecz.return_move_layers[0] == (compiler_lane.reverse(),)
    assert ecz.return_move_layers[1] == (user_lane.reverse(),)


# --- PalindromePlacementStrategy.cz_placements tests ---


def test_palindrome_cz_with_user_moved_produces_correct_execute_cz_return():
    from bloqade.lanes.analysis.placement.lattice import ExecuteCZReturn
    from bloqade.lanes.analysis.placement.strategy import PalindromePlacementStrategy

    inner = _make_strategy()
    strat = PalindromePlacementStrategy(inner=inner)

    # Home layout: qubit0 at word 0, qubit1 at word 2 (both home positions in logical arch)
    home_layout = (
        _loc(0, 0, 0),
        _loc(0, 2, 0),
    )
    # User moved qubit1 to word 1 — CZ entangling-pair partner of word 0
    dest_layout = (
        _loc(0, 0, 0),
        _loc(0, 1, 0),
    )
    src_state = _concrete(home_layout)
    target_state = _concrete(dest_layout)
    move_layers = inner.compute_moves(src_state, target_state)

    um = UserMoved.from_concrete_state(
        target_state,
        move_layers=move_layers,
        accumulated_move_layers=move_layers,
        pre_user_layout=home_layout,
    )

    result = strat.cz_placements(um, controls=(0,), targets=(1,))
    assert isinstance(result, ExecuteCZReturn)
    assert result.user_move_layers == um.accumulated_move_layers
    assert result.initial_layout == um.pre_user_layout


def test_palindrome_cz_without_user_moved_unchanged():
    from bloqade.lanes.analysis.placement.lattice import ExecuteCZReturn
    from bloqade.lanes.analysis.placement.strategy import PalindromePlacementStrategy

    inner = _make_strategy()
    strat = PalindromePlacementStrategy(inner=inner)

    # Plain ConcreteState: qubit0 at word 0, qubit1 at word 2
    layout = (
        _loc(0, 0, 0),
        _loc(0, 2, 0),
    )
    state = _concrete(layout)
    result = strat.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(result, ExecuteCZReturn)
    assert result.user_move_layers == ()  # no user moves involved


def test_palindrome_sq_placements_user_moved_returns_bottom():
    """PalindromePlacementStrategy rejects UserMoved before SQ gate."""
    from bloqade.lanes.analysis.placement.strategy import PalindromePlacementStrategy

    inner = _make_strategy()
    strat = PalindromePlacementStrategy(inner=inner)
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    um = UserMoved.from_concrete_state(
        _concrete(layout),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=layout,
    )
    result = strat.sq_placements(um, qubits=(0,))
    assert result == AtomState.bottom()
