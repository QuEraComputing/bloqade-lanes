from typing import Any, cast

from bloqade.native.dialects.gate import stmts as gates
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.analysis import const
from kirin.dialects import ilist, py

from bloqade import qubit
from bloqade.gemini.common.dialects.qubit import stmts as gemini_common_stmts
from bloqade.gemini.logical.dialects.operations import stmts as gemini_stmts
from bloqade.lanes import types
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.circuit2place import MergeStaticPlacement  # new
from bloqade.lanes.rewrite.circuit2place import always_merge  # new
from bloqade.lanes.rewrite.circuit2place import gate_only_merge  # new
from bloqade.lanes.rewrite.circuit2place import (
    InitializeNewQubits,
    MergePlacementRegions,
    RewriteLogicalInitializeToNewLogical,
    RewritePlaceOperations,
)


def test_cz():

    test_block = ir.Block(
        [
            targets := ilist.New(values=(q0 := ir.TestValue(), q1 := ir.TestValue())),
            controls := ilist.New(values=(c0 := ir.TestValue(), c1 := ir.TestValue())),
            gates.CZ(targets=targets.result, controls=controls.result),
        ],
    )

    expected_block = ir.Block(
        [
            targets := ilist.New(values=(q0, q1)),
            controls := ilist.New(values=(c0, c1)),
            place.StaticPlacement(
                qubits=(c0, c1, q0, q1), body=ir.Region(block := ir.Block())
            ),
        ]
    )

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(gate_stmt := place.CZ(entry_state, qubits=(0, 1, 2, 3)))
    block.stmts.append(place.Yield(gate_stmt.state_after))

    rule = rewrite.Walk(RewritePlaceOperations())

    rule.rewrite(test_block)

    assert_nodes(test_block, expected_block)


test_cz()


def test_r():
    axis_angle = ir.TestValue()
    rotation_angle = ir.TestValue()
    test_block = ir.Block(
        [
            inputs := ilist.New(values=(q0 := ir.TestValue(), q1 := ir.TestValue())),
            gates.R(
                qubits=inputs.result,
                axis_angle=axis_angle,
                rotation_angle=rotation_angle,
            ),
        ],
    )

    expected_block = ir.Block(
        [
            inputs := ilist.New(values=(q0, q1)),
            place.StaticPlacement(qubits=(q0, q1), body=ir.Region(block := ir.Block())),
        ]
    )

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(
        gate_stmt := place.R(
            entry_state,
            qubits=(0, 1),
            axis_angle=axis_angle,
            rotation_angle=rotation_angle,
        )
    )
    block.stmts.append(place.Yield(gate_stmt.state_after))

    rule = rewrite.Walk(RewritePlaceOperations())

    rule.rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_rz():
    rotation_angle = ir.TestValue()
    test_block = ir.Block(
        [
            qubits := ilist.New(values=(q0 := ir.TestValue(), q1 := ir.TestValue())),
            gates.Rz(qubits=qubits.result, rotation_angle=rotation_angle),
        ],
    )

    expected_block = ir.Block(
        [
            qubits := ilist.New(values=(q0, q1)),
            place.StaticPlacement(qubits=(q0, q1), body=ir.Region(block := ir.Block())),
        ]
    )

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(
        gate_stmt := place.Rz(entry_state, qubits=(0, 1), rotation_angle=rotation_angle)
    )
    block.stmts.append(place.Yield(gate_stmt.state_after))

    rule = rewrite.Walk(RewritePlaceOperations())

    rule.rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_star_rz():
    rotation_angle = ir.TestValue()
    test_block = ir.Block(
        [
            qubits := ilist.New(values=(q0 := ir.TestValue(), q1 := ir.TestValue())),
            # TODO: why is this cast as Any? is there a better way to get around this pyright check failure?
            cast(Any, gemini_stmts.StarRz)(
                rotation_angle=rotation_angle,
                qubits=qubits.result,
                qubit_indices=(0, 2, 4),
            ),
        ],
    )

    expected_block = ir.Block(
        [
            qubits := ilist.New(values=(q0, q1)),
            place.StaticPlacement(qubits=(q0, q1), body=ir.Region(block := ir.Block())),
        ]
    )

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(
        gate_stmt := place.StarRz(
            entry_state,
            rotation_angle,
            qubits=(0, 1),
            qubit_indices=(0, 2, 4),
        )
    )
    block.stmts.append(place.Yield(gate_stmt.state_after))

    rule = rewrite.Walk(RewritePlaceOperations())

    rule.rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_measurement():
    test_block = ir.Block(
        [
            qubits := ilist.New(
                values=(
                    q0 := ir.TestValue(),
                    q1 := ir.TestValue(),
                    q2 := ir.TestValue(),
                )
            ),
            gemini_stmts.TerminalLogicalMeasurement(qubits=qubits.result),
        ],
    )

    expected_block = ir.Block(
        [
            qubits := ilist.New(values=(q0, q1, q2)),
        ]
    )

    block = ir.Block()

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(gate_stmt := place.EndMeasure(entry_state, qubits=(0, 1, 2)))
    block.stmts.append(place.Yield(*gate_stmt.results))
    expected_block.stmts.append(
        circ := place.StaticPlacement(qubits=(q0, q1, q2), body=ir.Region(block))
    )
    expected_block.stmts.append(
        place.ConvertToPhysicalMeasurements(tuple(circ.results))
    )
    rule = rewrite.Walk(RewritePlaceOperations())

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_initialize():
    test_block = ir.Block(
        [
            qubits := ilist.New(
                values=(
                    q0 := ir.TestValue(),
                    q1 := ir.TestValue(),
                    q2 := ir.TestValue(),
                )
            ),
            gemini_stmts.Initialize(
                theta := ir.TestValue(),
                phi := ir.TestValue(),
                lam := ir.TestValue(),
                qubits=qubits.result,
            ),
        ],
    )

    expected_block = ir.Block(
        [
            qubits := ilist.New(values=(q0, q1, q2)),
        ]
    )

    block = ir.Block()

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(
        gate_stmt := place.Initialize(
            entry_state,
            theta=theta,
            phi=phi,
            lam=lam,
            qubits=(0, 1, 2),
        )
    )
    block.stmts.append(place.Yield(gate_stmt.state_after))
    expected_block.stmts.append(
        place.StaticPlacement(qubits=(q0, q1, q2), body=ir.Region(block))
    )
    rule = rewrite.Walk(RewritePlaceOperations())

    rule.rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_merge_regions():

    qubits = tuple(ir.TestValue() for _ in range(10))

    test_block = ir.Block([rotation_angle := py.Constant(0.5)])
    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, name="entry_state")
    body_block.stmts.append(
        gate_stmt := place.Rz(
            entry_state, qubits=(0, 1), rotation_angle=rotation_angle.result
        )
    )
    body_block.stmts.append(
        measure0_stmt := place.EndMeasure(gate_stmt.state_after, qubits=(0, 1))
    )
    body_block.stmts.append(place.Yield(*measure0_stmt.results))
    test_block.stmts.append(
        circuit1 := place.StaticPlacement(
            qubits=(qubits[0], qubits[1]), body=ir.Region(body_block)
        )
    )

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, name="entry_state")
    body_block.stmts.append(
        gate_stmt := place.Rz(
            entry_state, qubits=(0, 1), rotation_angle=rotation_angle.result
        )
    )
    body_block.stmts.append(
        measure1_stmt := place.EndMeasure(gate_stmt.state_after, qubits=(0, 1))
    )
    body_block.stmts.append(place.Yield(*measure1_stmt.results))
    test_block.stmts.append(
        circuit2 := place.StaticPlacement(
            qubits=(qubits[2], qubits[3]), body=ir.Region(body_block)
        )
    )

    test_block.stmts.append(
        ilist.New(tuple(circuit1.results) + tuple(circuit2.results))
    )

    expected_block = ir.Block([rotation_angle := py.Constant(0.5)])
    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, name="entry_state")
    body_block.stmts.append(
        (
            gate_stmt := place.Rz(
                entry_state, qubits=(0, 1), rotation_angle=rotation_angle.result
            )
        )
    )
    body_block.stmts.append(
        measure01_stmt := place.EndMeasure(gate_stmt.state_after, qubits=(0, 1))
    )
    body_block.stmts.append(
        gate_stmt := place.Rz(
            gate_stmt.state_after, qubits=(2, 3), rotation_angle=rotation_angle.result
        )
    )
    body_block.stmts.append(
        measure23_stmt := place.EndMeasure(gate_stmt.state_after, qubits=(2, 3))
    )
    measure_result = tuple(measure01_stmt.results[1:]) + tuple(
        measure23_stmt.results[1:]
    )
    body_block.stmts.append(
        place.Yield(
            gate_stmt.state_after,
            *measure_result,
        )
    )
    expected_block.stmts.append(
        merged_circuit := place.StaticPlacement(
            qubits=(qubits[0], qubits[1], qubits[2], qubits[3]),
            body=ir.Region(body_block),
        )
    )
    expected_block.stmts.append(ilist.New(tuple(merged_circuit.results)))

    rewrite.Fixpoint(rewrite.Walk(MergePlacementRegions())).rewrite(test_block)

    test_block.print()
    expected_block.print()
    assert_nodes(test_block, expected_block)


def _make_const_new_at(zone: int, word: int, site: int) -> gemini_common_stmts.NewAt:
    """Helper: build a NewAt whose three args carry const-prop hints."""
    c_zone = py.Constant(zone)
    c_word = py.Constant(word)
    c_site = py.Constant(site)
    c_zone.result.hints["const"] = const.Value(zone)
    c_word.result.hints["const"] = const.Value(word)
    c_site.result.hints["const"] = const.Value(site)
    new_at = gemini_common_stmts.NewAt(
        zone_id=c_zone.result, word_id=c_word.result, site_id=c_site.result
    )
    return new_at


def test_new_at_with_const_args_produces_pinned_new_logical_qubit():
    """NewAt with const args → place.NewLogicalQubit with location_address."""
    block = ir.Block()

    new_at = _make_const_new_at(zone=1, word=2, site=3)
    # insert the py.Constant owners into the block first
    for arg in (new_at.zone_id, new_at.word_id, new_at.site_id):
        block.stmts.append(arg.owner)  # type: ignore[arg-type]
    block.stmts.append(new_at)

    theta = ir.TestValue()
    phi = ir.TestValue()
    lam = ir.TestValue()
    init = place.LogicalInitialize(
        theta=theta, phi=phi, lam=lam, qubits=(new_at.qubit,)
    )
    block.stmts.append(init)

    rewrite.Walk(RewriteLogicalInitializeToNewLogical()).rewrite(block)

    # After the rewrite, new_at should be replaced by a NewLogicalQubit
    stmts = list(block.stmts)
    new_logical_qubits = [s for s in stmts if isinstance(s, place.NewLogicalQubit)]
    assert (
        len(new_logical_qubits) == 1
    ), f"Expected 1 NewLogicalQubit, got {len(new_logical_qubits)}"

    nq = new_logical_qubits[0]
    expected_addr = LocationAddress(word_id=2, site_id=3, zone_id=1)
    assert (
        nq.location_address == expected_addr
    ), f"Expected location_address={expected_addr!r}, got {nq.location_address!r}"


def test_mixed_kernel_new_and_new_at():
    """Both qubit.stmts.New and NewAt in the same LogicalInitialize are both rewritten."""
    block = ir.Block()

    # un-pinned qubit
    plain_new = qubit.stmts.New()
    block.stmts.append(plain_new)

    # pinned qubit via NewAt
    new_at = _make_const_new_at(zone=0, word=5, site=7)
    for arg in (new_at.zone_id, new_at.word_id, new_at.site_id):
        block.stmts.append(arg.owner)  # type: ignore[arg-type]
    block.stmts.append(new_at)

    theta = ir.TestValue()
    phi = ir.TestValue()
    lam = ir.TestValue()
    init = place.LogicalInitialize(
        theta=theta, phi=phi, lam=lam, qubits=(plain_new.result, new_at.qubit)
    )
    block.stmts.append(init)

    rewrite.Walk(RewriteLogicalInitializeToNewLogical()).rewrite(block)

    stmts = list(block.stmts)
    new_logical_qubits = [s for s in stmts if isinstance(s, place.NewLogicalQubit)]
    assert (
        len(new_logical_qubits) == 2
    ), f"Expected 2 NewLogicalQubits, got {len(new_logical_qubits)}"

    unpinned = [nq for nq in new_logical_qubits if nq.location_address is None]
    pinned = [nq for nq in new_logical_qubits if nq.location_address is not None]
    assert len(unpinned) == 1, "Expected exactly 1 un-pinned NewLogicalQubit"
    assert len(pinned) == 1, "Expected exactly 1 pinned NewLogicalQubit"

    expected_addr = LocationAddress(word_id=5, site_id=7, zone_id=0)
    assert pinned[0].location_address == expected_addr


def test_pure_qubit_new_regression():
    """All-qubit.stmts.New kernel produces no location_address (regression guard)."""
    block = ir.Block()

    plain_new = qubit.stmts.New()
    block.stmts.append(plain_new)

    theta = ir.TestValue()
    phi = ir.TestValue()
    lam = ir.TestValue()
    init = place.LogicalInitialize(
        theta=theta, phi=phi, lam=lam, qubits=(plain_new.result,)
    )
    block.stmts.append(init)

    rewrite.Walk(RewriteLogicalInitializeToNewLogical()).rewrite(block)

    stmts = list(block.stmts)
    new_logical_qubits = [s for s in stmts if isinstance(s, place.NewLogicalQubit)]
    assert len(new_logical_qubits) == 1
    assert new_logical_qubits[0].location_address is None


def test_new_at_with_non_const_args_is_noop():
    """NewAt with a non-constant arg is left in place (no crash, no replacement)."""
    block = ir.Block()

    # zone_id is a plain TestValue (no const hint) — simulates a function argument
    non_const_zone = ir.TestValue()
    c_word = py.Constant(0)
    c_site = py.Constant(0)
    c_word.result.hints["const"] = const.Value(0)
    c_site.result.hints["const"] = const.Value(0)
    block.stmts.append(c_word)
    block.stmts.append(c_site)

    new_at = gemini_common_stmts.NewAt(
        zone_id=non_const_zone, word_id=c_word.result, site_id=c_site.result
    )
    block.stmts.append(new_at)

    theta = ir.TestValue()
    phi = ir.TestValue()
    lam = ir.TestValue()
    init = place.LogicalInitialize(
        theta=theta, phi=phi, lam=lam, qubits=(new_at.qubit,)
    )
    block.stmts.append(init)

    # Should not raise; should not replace the NewAt
    rewrite.Walk(RewriteLogicalInitializeToNewLogical()).rewrite(block)

    stmts = list(block.stmts)
    # The NewAt should still be present (not replaced)
    new_ats = [s for s in stmts if isinstance(s, gemini_common_stmts.NewAt)]
    new_logical_qubits = [s for s in stmts if isinstance(s, place.NewLogicalQubit)]
    assert len(new_ats) == 1, "NewAt should remain when const-prop hint is missing"
    assert len(new_logical_qubits) == 0, "No NewLogicalQubit should be emitted"


# ---------------------------------------------------------------------------
# D3 tests — InitializeNewQubits handles bare NewAt (not wrapped in Initialize)
# ---------------------------------------------------------------------------


def test_initialize_new_qubits_bare_new_at_with_const_args():
    """Bare NewAt (no enclosing Initialize) with const args → pinned NewLogicalQubit."""
    block = ir.Block()

    new_at = _make_const_new_at(zone=2, word=4, site=6)
    for arg in (new_at.zone_id, new_at.word_id, new_at.site_id):
        block.stmts.append(arg.owner)  # type: ignore[arg-type]
    block.stmts.append(new_at)

    rewrite.Walk(InitializeNewQubits()).rewrite(block)

    stmts = list(block.stmts)
    new_logical_qubits = [s for s in stmts if isinstance(s, place.NewLogicalQubit)]
    assert (
        len(new_logical_qubits) == 1
    ), f"Expected 1 NewLogicalQubit, got {len(new_logical_qubits)}"

    nq = new_logical_qubits[0]
    expected_addr = LocationAddress(word_id=4, site_id=6, zone_id=2)
    assert (
        nq.location_address == expected_addr
    ), f"Expected location_address={expected_addr!r}, got {nq.location_address!r}"

    # The rewrite should have injected at least one py.Constant (the zero angle)
    constants = [s for s in stmts if isinstance(s, py.Constant)]
    assert (
        len(constants) >= 1
    ), "Expected a py.Constant to be injected for angle defaults"


def test_initialize_new_qubits_bare_qubit_new_regression():
    """Bare qubit.stmts.New → NewLogicalQubit with no location_address (regression guard)."""
    block = ir.Block()

    plain_new = qubit.stmts.New()
    block.stmts.append(plain_new)

    rewrite.Walk(InitializeNewQubits()).rewrite(block)

    stmts = list(block.stmts)
    new_logical_qubits = [s for s in stmts if isinstance(s, place.NewLogicalQubit)]
    assert len(new_logical_qubits) == 1
    assert new_logical_qubits[0].location_address is None


def test_initialize_new_qubits_bare_new_at_non_const_is_noop():
    """Bare NewAt with a non-constant arg is left in place (no crash, no replacement)."""
    block = ir.Block()

    non_const_zone = ir.TestValue()
    c_word = py.Constant(0)
    c_site = py.Constant(0)
    c_word.result.hints["const"] = const.Value(0)
    c_site.result.hints["const"] = const.Value(0)
    block.stmts.append(c_word)
    block.stmts.append(c_site)

    new_at = gemini_common_stmts.NewAt(
        zone_id=non_const_zone, word_id=c_word.result, site_id=c_site.result
    )
    block.stmts.append(new_at)

    # Should not raise; should not replace the NewAt
    rewrite.Walk(InitializeNewQubits()).rewrite(block)

    stmts = list(block.stmts)
    new_ats = [s for s in stmts if isinstance(s, gemini_common_stmts.NewAt)]
    new_logical_qubits = [s for s in stmts if isinstance(s, place.NewLogicalQubit)]
    assert len(new_ats) == 1, "NewAt should remain when const-prop hint is missing"
    assert len(new_logical_qubits) == 0, "No NewLogicalQubit should be emitted"


def test_merge_static_placement_always_merge():
    """MergeStaticPlacement(always_merge) merges two EndMeasure blocks (same as MergePlacementRegions)."""
    # Re-implementation of test_merge_regions using the new class.
    qubits = tuple(ir.TestValue() for _ in range(4))

    test_block = ir.Block([rotation_angle := py.Constant(0.5)])

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, name="entry_state")
    body_block.stmts.append(
        gate_stmt := place.Rz(
            entry_state, qubits=(0,), rotation_angle=rotation_angle.result
        )
    )
    body_block.stmts.append(place.Yield(gate_stmt.state_after))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[0],), body=ir.Region(body_block))
    )

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, name="entry_state")
    body_block.stmts.append(
        gate_stmt := place.Rz(
            entry_state, qubits=(0,), rotation_angle=rotation_angle.result
        )
    )
    body_block.stmts.append(place.Yield(gate_stmt.state_after))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[1],), body=ir.Region(body_block))
    )

    rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(always_merge))).rewrite(
        test_block
    )

    merged_stmts = [s for s in test_block.stmts if isinstance(s, place.StaticPlacement)]
    assert len(merged_stmts) == 1
    body_stmts = list(merged_stmts[0].body.blocks[0].stmts)
    # two Rz + one Yield
    assert len(body_stmts) == 3
    assert isinstance(body_stmts[0], place.Rz)
    assert isinstance(body_stmts[1], place.Rz)
    assert isinstance(body_stmts[2], place.Yield)


def test_gate_only_merge_allows_pure_gate_blocks():
    """gate_only_merge merges two placements that contain only R/Rz/CZ/Yield."""
    qubits = tuple(ir.TestValue() for _ in range(2))
    test_block = ir.Block([angle := py.Constant(0.5)])

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, "entry_state")
    body_block.stmts.append(
        g := place.Rz(entry_state, qubits=(0,), rotation_angle=angle.result)
    )
    body_block.stmts.append(place.Yield(g.state_after))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[0],), body=ir.Region(body_block))
    )

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, "entry_state")
    body_block.stmts.append(
        g := place.Rz(entry_state, qubits=(0,), rotation_angle=angle.result)
    )
    body_block.stmts.append(place.Yield(g.state_after))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[1],), body=ir.Region(body_block))
    )

    rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(gate_only_merge))).rewrite(
        test_block
    )

    merged_stmts = [s for s in test_block.stmts if isinstance(s, place.StaticPlacement)]
    assert len(merged_stmts) == 1


def test_gate_only_merge_rejects_initialize_block():
    """gate_only_merge does NOT merge a placement containing place.Initialize."""
    qubits = tuple(ir.TestValue() for _ in range(2))
    test_block = ir.Block(
        [
            theta := py.Constant(0.0),
            phi := py.Constant(0.0),
            lam := py.Constant(0.0),
            angle := py.Constant(0.5),
        ]
    )

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, "entry_state")
    body_block.stmts.append(
        init := place.Initialize(
            entry_state, theta=theta.result, phi=phi.result, lam=lam.result, qubits=(0,)
        )
    )
    body_block.stmts.append(place.Yield(init.state_after))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[0],), body=ir.Region(body_block))
    )

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, "entry_state")
    body_block.stmts.append(
        g := place.Rz(entry_state, qubits=(0,), rotation_angle=angle.result)
    )
    body_block.stmts.append(place.Yield(g.state_after))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[1],), body=ir.Region(body_block))
    )

    rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(gate_only_merge))).rewrite(
        test_block
    )

    # Must NOT be merged
    remaining = [s for s in test_block.stmts if isinstance(s, place.StaticPlacement)]
    assert len(remaining) == 2


def test_gate_only_merge_rejects_end_measure_block():
    """gate_only_merge does NOT merge a placement containing place.EndMeasure."""
    qubits = tuple(ir.TestValue() for _ in range(2))
    test_block = ir.Block([angle := py.Constant(0.5)])

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, "entry_state")
    body_block.stmts.append(em := place.EndMeasure(entry_state, qubits=(0,)))
    body_block.stmts.append(place.Yield(*em.results))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[0],), body=ir.Region(body_block))
    )

    body_block = ir.Block()
    entry_state = body_block.args.append_from(types.StateType, "entry_state")
    body_block.stmts.append(
        g := place.Rz(entry_state, qubits=(0,), rotation_angle=angle.result)
    )
    body_block.stmts.append(place.Yield(g.state_after))
    test_block.stmts.append(
        place.StaticPlacement(qubits=(qubits[1],), body=ir.Region(body_block))
    )

    rewrite.Fixpoint(rewrite.Walk(MergeStaticPlacement(gate_only_merge))).rewrite(
        test_block
    )

    remaining = [s for s in test_block.stmts if isinstance(s, place.StaticPlacement)]
    assert len(remaining) == 2
