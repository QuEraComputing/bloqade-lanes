from kirin import ir, rewrite
from kirin.dialects import func

from bloqade import qubit
from bloqade.lanes import kernel, layout
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.move2squin import base


def test_insert_qubits():
    location_addresses = (
        layout.LocationAddress(0, 0),
        layout.LocationAddress(0, 1),
    )

    @kernel
    def main():
        state = move.load()
        return move.fill(state, location_addresses=location_addresses)

    atom_interp = atom.AtomInterpreter(kernel, arch_spec=get_arch_spec())

    frame, _ = atom_interp.run(main)

    rewrite.Walk(base.InsertQubits(atom_state_map=frame)).rewrite(main.code)

    assert len(main.callable_region.blocks) == 1
    block = main.callable_region.blocks[0]

    assert isinstance(block.stmts.at(0), move.Load)
    assert isinstance(block.stmts.at(1), qubit.stmts.New)
    assert isinstance(block.stmts.at(2), qubit.stmts.New)
    assert isinstance(block.stmts.at(3), move.Fill)
    assert isinstance(block.stmts.at(4), func.Return)


def test_base_rewrite_rule():
    rule = base.AtomStateRewriter(
        arch_spec=get_arch_spec(),
        physical_ssa_values={0: (zero := ir.TestValue()), 1: (one := ir.TestValue())},
    )
    atom_state_data = atom.AtomStateData.new(
        {0: layout.LocationAddress(0, 0), 1: layout.LocationAddress(0, 1)}
    )
    atom_state = atom.AtomState(atom_state_data)

    locations = (
        layout.LocationAddress(0, 0),
        layout.LocationAddress(0, 1),
        layout.LocationAddress(1, 0),
    )
    assert rule.get_qubit_ssa_from_locations(atom_state, locations) == (zero, one, None)


def test_remove_move():
    @kernel
    def test():
        move.load()
        return

    rewrite.Walk(base.CleanUpMoveDialect()).rewrite(test.code)

    block = test.callable_region.blocks[0]

    assert len(block.stmts) == 2
    assert isinstance(block.stmts.at(0), func.ConstantNone)
    assert isinstance(block.stmts.at(1), func.Return)


if __name__ == "__main__":
    test_remove_move()
