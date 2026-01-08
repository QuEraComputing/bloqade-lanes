from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.analysis import forward
from kirin.dialects import ilist

from bloqade import squin
from bloqade.lanes import kernel, layout
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec, steane7_initialize
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.move2squin import base, gates


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
            initialize_kernel=steane7_initialize,
        )
    ).rewrite(test_kernel.code)


def test_gate_rewrite():
    state = ir.TestValue()

    test_block = ir.Block(
        [move_CZ := move.CZ(current_state=state, zone_address=layout.ZoneAddress(0))]
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
        move_CZ, entries={move_CZ.result: atom_state}
    )

    rule = gates.InsertGates(
        arch_spec=arch_spec,
        physical_ssa_values=physical_ssa_values,  # type: ignore
        move_exec_analysis=frame,
        initialize_kernel=steane7_initialize,
    )
    rewrite.Walk(rule).rewrite(test_block)

    expected_block = ir.Block(
        [
            ctrl := ilist.New((zero,)),
            trgt := ilist.New((one,)),
            squin.gate.stmts.CZ(controls=ctrl.result, targets=trgt.result),
            move.CZ(current_state=state, zone_address=layout.ZoneAddress(0)),
        ]
    )

    assert_nodes(test_block, expected_block)


if __name__ == "__main__":
    test_gate_rewrite()
