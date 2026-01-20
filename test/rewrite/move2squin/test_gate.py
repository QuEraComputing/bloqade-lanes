from typing import Any

from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.analysis import forward
from kirin.dialects import func, ilist, py

from bloqade import qubit, squin, types as bloqade_types
from bloqade.lanes import kernel, layout
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.move2squin import base, gates


@squin.kernel
def initialize(
    theta: float, phi: float, lam: float, qubits: ilist.IList[bloqade_types.Qubit, Any]
):
    return


def run_insert_qubits(test_kernel: ir.Method):

    arch_spec = get_arch_spec()

    atom_interp = atom.AtomInterpreter(kernel, arch_spec=arch_spec)

    frame, _ = atom_interp.run(test_kernel)
    rule = base.InsertQubits(atom_state_map=frame)
    rewrite.Walk(rule).rewrite(test_kernel.code)
    rewrite.Walk(
        gates.InsertGates(
            arch_spec=arch_spec,
            physical_ssa_values=rule.physical_ssa_values,
            move_exec_analysis=frame,
            initialize_kernel=initialize,
        )
    ).rewrite(test_kernel.code)


def test_gate_rewrite_cz():
    state = ir.TestValue()

    test_block = ir.Block(
        [gate_node := move.CZ(current_state=state, zone_address=layout.ZoneAddress(0))]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    arch_spec = get_arch_spec()
    atom_state = atom.AtomState(
        atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 5),
            }
        )
    )
    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            ctrl := ilist.New((zero,)),
            trgt := ilist.New((one,)),
            squin.gate.stmts.CZ(controls=ctrl.result, targets=trgt.result),
            gate_node,
        ]
    )

    assert_nodes(test_block, expected_block)


def test_gate_rewrite_global_rz():
    state = ir.TestValue()
    rotation_angle = ir.TestValue()
    test_block = ir.Block(
        [gate_node := move.GlobalRz(current_state=state, rotation_angle=rotation_angle)]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    arch_spec = get_arch_spec()
    atom_state = atom.AtomState(
        atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 5),
            }
        )
    )
    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            const_zero := py.Constant(0.0),
            reg := ilist.New((zero, one)),
            squin.gate.stmts.U3(
                const_zero.result, rotation_angle, const_zero.result, reg.result
            ),
            gate_node,
        ]
    )

    assert_nodes(test_block, expected_block)


def test_gate_rewrite_global_r():
    state = ir.TestValue()
    rotation_angle = ir.TestValue()
    axis_angle = ir.TestValue()
    test_block = ir.Block(
        [
            gate_node := move.GlobalR(
                current_state=state,
                rotation_angle=rotation_angle,
                axis_angle=axis_angle,
            )
        ]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    arch_spec = get_arch_spec()
    atom_state = atom.AtomState(
        atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 5),
            }
        )
    )
    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            const_quarter := py.Constant(0.25),
            phi := py.Sub(const_quarter.result, axis_angle),
            lam := py.Sub(axis_angle, const_quarter.result),
            reg := ilist.New((zero, one)),
            squin.gate.stmts.U3(rotation_angle, phi.result, lam.result, reg.result),
            move.GlobalR(
                current_state=state,
                rotation_angle=rotation_angle,
                axis_angle=axis_angle,
            ),
        ]
    )
    assert_nodes(test_block, expected_block)


def test_gate_rewrite_local_r():
    state = ir.TestValue()
    rotation_angle = ir.TestValue()
    axis_angle = ir.TestValue()
    test_block = ir.Block(
        [
            gate_node := move.LocalR(
                current_state=state,
                rotation_angle=rotation_angle,
                axis_angle=axis_angle,
                location_addresses=(
                    layout.LocationAddress(0, 0),
                    layout.LocationAddress(0, 5),
                ),
            )
        ]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    arch_spec = get_arch_spec()
    atom_state = atom.AtomState(
        atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 5),
            }
        )
    )
    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            const_quarter := py.Constant(0.25),
            phi := py.Sub(const_quarter.result, axis_angle),
            lam := py.Sub(axis_angle, const_quarter.result),
            reg := ilist.New((zero, one)),
            squin.gate.stmts.U3(rotation_angle, phi.result, lam.result, reg.result),
            gate_node,
        ]
    )
    assert_nodes(test_block, expected_block)


def test_gate_rewrite_local_rz():
    state = ir.TestValue()
    rotation_angle = ir.TestValue()
    test_block = ir.Block(
        [
            gate_node := move.LocalRz(
                current_state=state,
                rotation_angle=rotation_angle,
                location_addresses=(
                    layout.LocationAddress(0, 0),
                    layout.LocationAddress(0, 5),
                ),
            )
        ]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    arch_spec = get_arch_spec()
    atom_state = atom.AtomState(
        atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 5),
            }
        )
    )
    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            const_zero := py.Constant(0.0),
            reg := ilist.New((zero, one)),
            squin.gate.stmts.U3(
                const_zero.result, rotation_angle, const_zero.result, reg.result
            ),
            gate_node,
        ]
    )
    assert_nodes(test_block, expected_block)


def test_gate_rewrite_logical_initialize():
    state = ir.TestValue()
    theta = ir.TestValue()
    phi = ir.TestValue()
    lam = ir.TestValue()
    test_block = ir.Block(
        [
            gate_node := move.LogicalInitialize(
                current_state=state,
                thetas=(theta, theta),
                phis=(phi, phi),
                lams=(lam, lam),
                location_addresses=(
                    layout.LocationAddress(0, 0),
                    layout.LocationAddress(0, 5),
                ),
            )
        ]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    arch_spec = get_arch_spec()
    atom_state = atom.AtomState(
        atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 5),
            }
        )
    )
    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            zero_reg := ilist.New((zero,)),
            squin.gate.stmts.U3(theta, phi, lam, zero_reg.result),
            one_reg := ilist.New((one,)),
            squin.gate.stmts.U3(theta, phi, lam, one_reg.result),
            gate_node,
        ]
    )
    assert_nodes(test_block, expected_block)


def test_gate_rewrite_physical_initialize():
    state = ir.TestValue()
    theta = ir.TestValue()
    phi = ir.TestValue()
    lam = ir.TestValue()
    test_block = ir.Block(
        [
            gate_node := move.PhysicalInitialize(
                current_state=state,
                thetas=(theta, theta),
                phis=(phi, phi),
                lams=(lam, lam),
                location_addresses=(
                    (layout.LocationAddress(0, 0),),
                    (layout.LocationAddress(0, 5),),
                ),
            )
        ]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
        1: (one := ir.TestValue()),
    }
    arch_spec = get_arch_spec()
    atom_state = atom.AtomState(
        atom.AtomStateData.new(
            {
                0: layout.LocationAddress(0, 0),
                1: layout.LocationAddress(0, 5),
            }
        )
    )
    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            zero_reg := ilist.New((zero,)),
            func.Invoke(
                (theta, phi, lam, zero_reg.result),
                callee=initialize,
            ),
            one_reg := ilist.New((one,)),
            func.Invoke(
                (theta, phi, lam, one_reg.result),
                callee=initialize,
            ),
            gate_node,
        ]
    )
    assert_nodes(test_block, expected_block)


def test_insert_measurements():

    future = ir.TestValue()
    test_block = ir.Block(
        [
            gate_node := move.GetFutureResult(
                future,
                zone_address=layout.ZoneAddress(0),
                location_address=layout.LocationAddress(0, 0),
            )
        ]
    )

    physical_ssa_values = {
        0: (zero := ir.TestValue()),
    }

    frame: forward.ForwardFrame[atom.MoveExecution] = forward.ForwardFrame(
        gate_node, entries={gate_node.result: atom.MeasureResult(0)}
    )

    rule = gates.InsertMeasurements(
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            func.Invoke((zero,), callee=qubit.measure),
        ]
    )

    assert_nodes(test_block, expected_block)
