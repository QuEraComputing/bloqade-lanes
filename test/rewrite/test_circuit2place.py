from bloqade.gemini.dialects.logical import stmts as gemini_stmts
from bloqade.native.dialects.gate import stmts as gates
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade.lanes import types
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.circuit2place import (
    MergePlacementRegions,
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
