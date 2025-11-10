from bloqade.native.dialects.gate import stmts as gates
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade.lanes import types
from bloqade.lanes.dialects import circuit, execute
from bloqade.lanes.rewrite.native2circuit import (
    MergePlacementRegions,
    RewriteLowLevelCircuit,
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
            execute.ExecuteLowLevel(
                qubits=(c0, c1, q0, q1), body=ir.Region(block := ir.Block())
            ),
        ]
    )

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(
        gate_stmt := circuit.CZ(entry_state, controls=(0, 1), targets=(2, 3))
    )
    block.stmts.append(execute.ExitLowLevel(state=gate_stmt.state_after))

    rule = rewrite.Walk(RewriteLowLevelCircuit())

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
            execute.ExecuteLowLevel(
                qubits=(q0, q1), body=ir.Region(block := ir.Block())
            ),
        ]
    )

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(
        gate_stmt := circuit.R(
            entry_state,
            qubits=(0, 1),
            axis_angle=axis_angle,
            rotation_angle=rotation_angle,
        )
    )
    block.stmts.append(execute.ExitLowLevel(state=gate_stmt.state_after))

    rule = rewrite.Walk(RewriteLowLevelCircuit())

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
            execute.ExecuteLowLevel(
                qubits=(q0, q1), body=ir.Region(block := ir.Block())
            ),
        ]
    )

    entry_state = block.args.append_from(types.StateType, name="entry_state")
    block.stmts.append(
        gate_stmt := circuit.Rz(
            entry_state, qubits=(0, 1), rotation_angle=rotation_angle
        )
    )
    block.stmts.append(execute.ExitLowLevel(state=gate_stmt.state_after))

    rule = rewrite.Walk(RewriteLowLevelCircuit())

    rule.rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_merge_regions():

    qubits = tuple(ir.TestValue() for _ in range(10))

    test_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            execute.ExecuteLowLevel(
                qubits=(qubits[0], qubits[1]),
                body=ir.Region(body_block := ir.Block()),
            ),
            execute.ExecuteLowLevel(
                qubits=(qubits[2], qubits[3]),
                body=ir.Region(second_block := ir.Block()),
            ),
        ]
    )

    entry_state = body_block.args.append_from(types.StateType, name="entry_state")
    body_block.stmts.append(
        gate_stmt := circuit.Rz(
            entry_state, qubits=(0, 1), rotation_angle=rotation_angle.result
        )
    )
    body_block.stmts.append(execute.ExitLowLevel(state=gate_stmt.state_after))

    entry_state = second_block.args.append_from(types.StateType, name="entry_state")
    second_block.stmts.append(
        gate_stmt := circuit.Rz(
            entry_state, qubits=(0, 1), rotation_angle=rotation_angle.result
        )
    )
    second_block.stmts.append(execute.ExitLowLevel(state=gate_stmt.state_after))

    expected_block = ir.Block(
        [
            rotation_angle := py.Constant(0.5),
            execute.ExecuteLowLevel(
                qubits=(qubits[0], qubits[1], qubits[2], qubits[3]),
                body=ir.Region(body_block := ir.Block()),
            ),
        ]
    )

    entry_state = body_block.args.append_from(types.StateType, name="entry_state")
    body_block.stmts.append(
        (
            gate_stmt := circuit.Rz(
                entry_state, qubits=(0, 1), rotation_angle=rotation_angle.result
            )
        )
    )
    body_block.stmts.append(
        gate_stmt := circuit.Rz(
            gate_stmt.state_after, qubits=(2, 3), rotation_angle=rotation_angle.result
        )
    )
    body_block.stmts.append(execute.ExitLowLevel(state=gate_stmt.state_after))

    rewrite.Walk(MergePlacementRegions()).rewrite(test_block)

    assert_nodes(test_block, expected_block)
