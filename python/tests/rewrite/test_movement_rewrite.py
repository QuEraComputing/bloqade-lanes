"""Tests for movement.MoveTo → place.StaticPlacement(place.MoveTo) rewrite."""

from kirin import ir, rewrite
from kirin.analysis import const
from kirin.dialects import ilist, py

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import movement, place
from bloqade.lanes.rewrite.circuit2place import RewritePlaceOperations


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

    stmt = movement.MoveTo(qubits_new.result, locs_new.result, multi_move_warning=False)

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

    stmt = movement.MoveTo(qubits_new.result, locs_new.result)
    test_block = ir.Block([qubits_new, locs_new, stmt])

    _make_rule().rewrite(test_block)

    remaining = list(test_block.stmts)
    movement_stmts = [s for s in remaining if isinstance(s, movement.MoveTo)]
    assert len(movement_stmts) == 1  # unchanged
