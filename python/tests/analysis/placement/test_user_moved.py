"""Tests for UserMoved lattice element and ExecuteCZReturn user_move_layers extension."""

import pytest

from bloqade.lanes.analysis.placement.exceptions import PlacementError
from bloqade.lanes.analysis.placement.lattice import (
    ConcreteState,
    ExecuteCZReturn,
    ExecuteMeasure,
    UserMoved,
)
from bloqade.lanes.analysis.placement.strategy import PalindromePlacementStrategy
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


def test_move_to_placements_occupancy_conflict_raises():
    strat = _make_strategy()
    layout = (
        _loc(0, 0, 0),
        _loc(0, 2, 0),
    )
    state = _concrete(layout)
    # Try to move qubit 0 into qubit 1's current slot — conflict
    dest = _loc(0, 2, 0)
    with pytest.raises(PlacementError, match="held by unmoved qubit"):
        strat.move_to_placements(state, qubits=(0,), locations=(dest,))


def test_sq_placements_user_moved_strips_to_concrete_state():
    """Non-palindrome SQ gates strip UserMoved to plain ConcreteState."""
    strat = _make_strategy()
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    um = UserMoved.from_concrete_state(
        _concrete(layout),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=layout,
    )
    result = strat.sq_placements(um, qubits=(0,))
    assert isinstance(result, ConcreteState)
    assert not isinstance(result, UserMoved)  # stripped to plain ConcreteState


def test_measure_placements_user_moved_concretizes():
    """Non-palindrome strategies let a user-move flow through measurement:
    the UserMoved is treated as a concrete state and measured at its moved
    layout. The compiler tracks the final atom positions (single zone)."""
    strat = _make_strategy()
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    um = UserMoved.from_concrete_state(
        _concrete(layout),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=layout,
    )
    result = strat.measure_placements(um, qubits=(0, 1))
    assert isinstance(result, ExecuteMeasure)
    assert result.layout == layout


def test_palindrome_measure_placements_user_moved_raises():
    """Under PalindromePlacementStrategy a user-move must NOT reach a
    measurement: only a CZ commits a user-move (palindrome stage + return), so
    a UserMoved at measurement is invalid."""
    strat = PalindromePlacementStrategy(inner=_make_strategy())
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    um = UserMoved.from_concrete_state(
        _concrete(layout),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=layout,
    )
    with pytest.raises(PlacementError, match="pending user-directed move"):
        strat.measure_placements(um, qubits=(0, 1))


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


def test_palindrome_sq_placements_user_moved_preserves_state():
    """PalindromePlacementStrategy preserves UserMoved through SQ gates."""
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
    assert result is um  # UserMoved preserved so CZ can see accumulated move layers


# --- Regression: UserMoved must survive an intervening relabel-only Permute
# under palindrome, so the CZ that commits the segment still palindromes back
# through the user moves (Review finding 1). ------------------------------


def test_palindrome_permute_relabel_preserves_user_moved_history():
    """Under palindrome, ``permute(insert_moves=False)`` on a ``UserMoved`` input
    must not drop the user-move history: the next CZ has to palindrome the
    atoms all the way home, not stop at the mid-segment permuted layout.

    Currently ``permute_placements`` returns a bare ``ConcreteState`` when the
    input is ``UserMoved``, silently dropping ``accumulated_move_layers`` and
    ``pre_user_layout``. That makes the palindrome return skip the user MoveTo
    reverse leg — atoms end up physically at the moved position while analysis
    believes they returned to the permuted layout.

    The relabel does not move atoms and it does not add lane-indexed physical
    moves, so ``accumulated_move_layers`` is preserved verbatim. But
    ``pre_user_layout`` is qubit-id indexed, so it must be permuted the same
    way ``layout`` is — otherwise the qubit-id-to-atom mapping after palindrome
    return would be inconsistent with the relabel.
    """
    from bloqade.lanes.analysis.placement.strategy import PalindromePlacementStrategy

    inner = _make_strategy()
    strat = PalindromePlacementStrategy(inner=inner)

    home_layout = (_loc(0, 0, 0), _loc(0, 2, 0), _loc(0, 4, 0))

    # User moves qubit 2 to word 6, then relabel-swaps qubits 0/1. Qubits 0/1
    # stay at their home slots physically; the relabel is just a reference
    # permutation. The pending physical MoveTo of qubit 2 must survive.
    um = strat.move_to_placements(
        _concrete(home_layout), qubits=(2,), locations=(_loc(0, 6, 0),)
    )
    assert isinstance(um, UserMoved)
    accumulated = um.accumulated_move_layers

    permuted = strat.permute_placements(
        um, qubits=(0, 1), permutation=(1, 0), insert_moves=False
    )

    # The user-move history must not be lost. A silent downgrade to plain
    # ConcreteState is incorrect under palindrome — the next CZ would not
    # unwind the MoveTo.
    assert isinstance(permuted, UserMoved), (
        "permute(relabel-only) on UserMoved under palindrome dropped user-move "
        "history to a plain ConcreteState; palindrome return will skip the "
        "MoveTo reverse leg."
    )
    assert permuted.accumulated_move_layers == accumulated
    # pre_user_layout is qubit-id indexed and must permute with the layout:
    # qubits 0/1 swap, qubit 2 unaffected.
    assert permuted.pre_user_layout == (home_layout[1], home_layout[0], home_layout[2])


def test_palindrome_cz_after_permute_after_move_to_returns_all_the_way_home():
    """End-to-end analysis check for the same composition: after MoveTo then
    ``permute(insert_moves=False)`` then CZ, the resulting ``ExecuteCZReturn``
    must carry the original user moves and set ``initial_layout`` to the
    pre-MoveTo home. Otherwise the palindrome return is a no-op for the user
    leg and atoms are stranded at their moved positions."""
    from bloqade.lanes.analysis.placement.lattice import ExecuteCZReturn
    from bloqade.lanes.analysis.placement.strategy import PalindromePlacementStrategy

    inner = _make_strategy()
    strat = PalindromePlacementStrategy(inner=inner)

    home_layout = (_loc(0, 0, 0), _loc(0, 2, 0))

    # MoveTo qubit 1 → word 1 (the CZ partner of word 0), then relabel-swap.
    um = strat.move_to_placements(
        _concrete(home_layout), qubits=(1,), locations=(_loc(0, 1, 0),)
    )
    assert isinstance(um, UserMoved)
    user_moves = um.accumulated_move_layers

    permuted = strat.permute_placements(
        um, qubits=(0, 1), permutation=(1, 0), insert_moves=False
    )

    result = strat.cz_placements(permuted, controls=(0,), targets=(1,))
    assert isinstance(result, ExecuteCZReturn)
    assert result.user_move_layers == user_moves, (
        "ExecuteCZReturn lost the user-move history across an intervening "
        "relabel Permute; palindrome return will not unwind the MoveTo."
    )
    # initial_layout is qubit-id indexed and must reflect the intervening
    # relabel — atoms return to home physically, but qubits 0/1 swap labels.
    assert result.initial_layout == (home_layout[1], home_layout[0]), (
        "ExecuteCZReturn.initial_layout should be the pre-MoveTo home with "
        "qubit references permuted to match the intervening relabel Permute."
    )


# --- Regression: MoveTo/permute destination collisions must return bottom,
# not raise AssertionError from ConcreteState.__post_init__
# (Review findings 3 and 4). ----------------------------------------------


def test_move_to_placements_target_in_occupied_raises():
    """A MoveTo whose destination overlaps ``state.occupied`` (an external atom
    outside the static circuit) must be rejected as a placement conflict —
    raising ``PlacementError`` — rather than tripping the ConcreteState
    invariant that ``layout`` and ``occupied`` are disjoint."""
    strat = _make_strategy()
    external = _loc(0, 4, 0)
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    state = ConcreteState(
        occupied=frozenset({external}),
        layout=layout,
        move_count=(0, 0),
    )
    with pytest.raises(PlacementError, match="occupied by an external atom"):
        strat.move_to_placements(state, qubits=(0,), locations=(external,))


def test_move_to_placements_duplicate_destinations_raises():
    """A MoveTo call that names the same destination for two moving qubits
    would place two atoms at the same slot. The placement analysis must reject
    this by raising PlacementError rather than assert-fail inside ConcreteState."""
    strat = _make_strategy()
    layout = (_loc(0, 0, 0), _loc(0, 2, 0))
    state = _concrete(layout)
    dup = _loc(0, 4, 0)
    with pytest.raises(PlacementError, match="same destination"):
        strat.move_to_placements(state, qubits=(0, 1), locations=(dup, dup))


# NOTE: There is no permute analog for the ``occupied``-overlap MoveTo test.
# ``permute_placements`` derives destinations exclusively from ``state.layout``
# (via ``_resolve_permute_locations``), and ``ConcreteState.__post_init__``
# asserts ``layout`` is disjoint from ``occupied``, so the collision is
# unreachable by construction. Duplicate destinations are likewise unreachable
# because ``circuit2place.rewrite_Permute`` validates ``perm`` is a true
# permutation before placement analysis sees it.
