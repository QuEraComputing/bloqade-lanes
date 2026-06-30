"""Tests for place.UserMoveTo → place.StaticPlacement(place.MoveTo) rewrite."""

import warnings

from kirin import ir, rewrite
from kirin.analysis import const
from kirin.dialects import ilist, py

from bloqade.gemini.common.dialects import movement
from bloqade.lanes.analysis.placement.lattice import (
    AtomState,
    ConcreteState,
    UserMoved,
)
from bloqade.lanes.bytecode.encoding import Direction, LocationAddress, SiteLaneAddress
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.circuit2place import RewritePlaceOperations

_loc = lambda z, w, s: LocationAddress(zone_id=z, word_id=w, site_id=s)  # noqa: E731
_lane = lambda z, w, s, d=Direction.FORWARD: SiteLaneAddress(z, w, s, d)  # noqa: E731


def _make_user_moved(home_layout, dest_layout, layers):
    return UserMoved.from_concrete_state(
        ConcreteState(
            occupied=frozenset(),
            layout=dest_layout,
            move_count=(0,) * len(dest_layout),
        ),
        move_layers=layers,
        accumulated_move_layers=layers,
        pre_user_layout=home_layout,
    )


def _make_rule():
    return rewrite.Walk(RewritePlaceOperations())


def test_rewrite_move_to_produces_static_placement():
    """movement.MoveTo with const-foldable locations rewrites to place.StaticPlacement(place.MoveTo)."""
    q0 = ir.TestValue()
    q1 = ir.TestValue()
    qubits_new = ilist.New(values=(q0, q1))

    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_b = LocationAddress(zone_id=0, word_id=0, site_id=1)
    loc_a_val = py.Constant(loc_a)
    loc_b_val = py.Constant(loc_b)
    locs_new = ilist.New(values=(loc_a_val.result, loc_b_val.result))
    # Stamp const hint on the locations SSA value
    locs_new.result.hints["const"] = const.Value((loc_a, loc_b))

    stmt = movement.stmts.MoveTo(
        qubits_new.result, locs_new.result, multi_move_warning=False
    )

    test_block = ir.Block([qubits_new, loc_a_val, loc_b_val, locs_new, stmt])

    _make_rule().rewrite(test_block)

    remaining = list(test_block.stmts)
    static_placements = [s for s in remaining if isinstance(s, place.StaticPlacement)]
    assert len(static_placements) == 1
    inner_stmts = list(static_placements[0].body.blocks[0].stmts)
    move_to_stmts = [s for s in inner_stmts if isinstance(s, place.MoveTo)]
    assert len(move_to_stmts) == 1
    mt = move_to_stmts[0]
    assert mt.qubits == (0, 1)
    assert mt.locations == (loc_a, loc_b)
    assert mt.multi_move_warning is False


def test_rewrite_move_to_gives_up_without_const_hint():
    """movement.MoveTo without a const hint on locations gives up silently."""
    q0 = ir.TestValue()
    qubits_new = ilist.New(values=(q0,))
    loc_val = ir.TestValue()
    locs_new = ilist.New(values=(loc_val,))
    # No const hint on locs_new.result

    stmt = movement.stmts.MoveTo(qubits_new.result, locs_new.result)
    test_block = ir.Block([qubits_new, locs_new, stmt])

    _make_rule().rewrite(test_block)

    remaining = list(test_block.stmts)
    move_stmts = [s for s in remaining if isinstance(s, movement.stmts.MoveTo)]
    assert len(move_stmts) == 1  # unchanged


def test_rewrite_move_to_length_mismatch_raises():
    """A const-folded move_to whose location count != qubit count fails loudly,
    so it cannot silently miscompile when MoveToValidation is skipped."""
    import pytest

    q0 = ir.TestValue()
    q1 = ir.TestValue()
    qubits_new = ilist.New(values=(q0, q1))  # 2 qubits

    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_a_val = py.Constant(loc_a)
    locs_new = ilist.New(values=(loc_a_val.result,))  # only 1 location
    locs_new.result.hints["const"] = const.Value((loc_a,))

    stmt = movement.stmts.MoveTo(qubits_new.result, locs_new.result)
    test_block = ir.Block([qubits_new, loc_a_val, locs_new, stmt])

    with pytest.raises(ValueError, match="number of locations"):
        _make_rule().rewrite(test_block)


def test_rewrite_permute_invalid_perm_raises():
    """A const-folded permute whose perm is not a permutation of
    range(len(qubits)) fails loudly rather than miscompiling."""
    import pytest

    q0 = ir.TestValue()
    q1 = ir.TestValue()
    qubits_new = ilist.New(values=(q0, q1))  # 2 qubits

    bad_perm_val = py.Constant((0, 0))  # not a permutation of range(2)
    perm_new = ilist.New(values=(bad_perm_val.result,))
    perm_new.result.hints["const"] = const.Value((0, 0))

    stmt = movement.stmts.Permute(qubits_new.result, perm_new.result)
    test_block = ir.Block([qubits_new, bad_perm_val, perm_new, stmt])

    with pytest.raises(ValueError, match="not a permutation"):
        _make_rule().rewrite(test_block)


# ---------------------------------------------------------------------------
# InsertMoves + RewriteGates tests for place.MoveTo
# ---------------------------------------------------------------------------


def test_insert_moves_emits_forward_for_user_moved():
    """InsertMoves inserts Load/Move/Store before place.MoveTo but leaves the node."""
    from bloqade.lanes.rewrite import place2move

    home = (_loc(0, 0, 0), _loc(0, 0, 1))
    dest = (_loc(0, 1, 0), _loc(0, 0, 1))
    layers = ((_lane(0, 0, 0),),)
    um = _make_user_moved(home, dest, layers)

    state_before = ir.TestValue()
    mt_stmt = place.MoveTo(state_before, qubits=(0,), locations=(_loc(0, 1, 0),))
    analysis: dict[ir.SSAValue, AtomState] = {mt_stmt.state_after: um}
    test_block = ir.Block([mt_stmt])

    rewrite.Walk(place2move.InsertMoves(analysis)).rewrite(test_block)

    stmts = list(test_block.stmts)
    assert any(isinstance(s, place.MoveTo) for s in stmts)


def test_multi_move_warning_emitted():
    """InsertMoves emits UserWarning when multi_move_warning=True and layers > 1."""
    from bloqade.lanes.rewrite import place2move

    home = (_loc(0, 0, 0),)
    dest = (_loc(0, 1, 0),)
    layers = ((_lane(0, 0, 0),), (_lane(0, 0, 1),))  # 2 layers
    um = _make_user_moved(home, dest, layers)

    state_before = ir.TestValue()
    mt_stmt = place.MoveTo(
        state_before,
        qubits=(0,),
        locations=(_loc(0, 1, 0),),
        multi_move_warning=True,
    )
    analysis: dict[ir.SSAValue, AtomState] = {mt_stmt.state_after: um}
    test_block = ir.Block([mt_stmt])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rewrite.Walk(place2move.InsertMoves(analysis)).rewrite(test_block)

    assert any(issubclass(warning.category, UserWarning) for warning in w)


def test_multi_move_warning_suppressed():
    """InsertMoves does not emit a warning when multi_move_warning=False."""
    from bloqade.lanes.rewrite import place2move

    home = (_loc(0, 0, 0),)
    dest = (_loc(0, 1, 0),)
    layers = ((_lane(0, 0, 0),), (_lane(0, 0, 1),))
    um = _make_user_moved(home, dest, layers)

    state_before = ir.TestValue()
    mt_stmt = place.MoveTo(
        state_before,
        qubits=(0,),
        locations=(_loc(0, 1, 0),),
        multi_move_warning=False,
    )
    analysis: dict[ir.SSAValue, AtomState] = {mt_stmt.state_after: um}
    test_block = ir.Block([mt_stmt])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rewrite.Walk(place2move.InsertMoves(analysis)).rewrite(test_block)

    assert not any(issubclass(warning.category, UserWarning) for warning in w)


def test_rewrite_gates_deletes_place_move_to():
    """RewriteGates deletes place.MoveTo nodes (returns empty stmts list)."""
    from bloqade.lanes.rewrite import place2move

    home = (_loc(0, 0, 0),)
    dest = (_loc(0, 1, 0),)
    layers = ((_lane(0, 0, 0),),)
    um = _make_user_moved(home, dest, layers)

    state_before = ir.TestValue()
    mt_stmt = place.MoveTo(state_before, qubits=(0,), locations=(_loc(0, 1, 0),))
    analysis: dict[ir.SSAValue, AtomState] = {mt_stmt.state_after: um}
    test_block = ir.Block([mt_stmt])

    rewrite.Walk(place2move.RewriteGates(analysis)).rewrite(test_block)

    remaining = list(test_block.stmts)
    assert not any(isinstance(s, place.MoveTo) for s in remaining)
