from bloqade.native.upstream import SquinToNative
from kirin.dialects import ilist
from matplotlib import pyplot as plt

from bloqade import qubit, squin
from bloqade.lanes.analysis.atom.analysis import (
    AtomInterpreter,
    AtomState,
)
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.layout.encoding import MoveType
from bloqade.lanes.upstream import CircuitToMove, NativeToCircuit


@squin.kernel(typeinfer=True, fold=True)
def log_depth_ghz():
    size = 10
    q0 = qubit.new()
    # squin.h(q0)
    reg = ilist.IList([q0])
    for i in range(size):
        current = len(reg)
        missing = size - current
        if missing > current:
            num_alloc = current
        else:
            num_alloc = missing

        if num_alloc > 0:
            new_qubits = qubit.qalloc(num_alloc)
            squin.broadcast.cz(reg[:num_alloc], new_qubits)
            reg = reg + new_qubits


log_depth_ghz = SquinToNative().emit(log_depth_ghz)
log_depth_ghz = NativeToCircuit().emit(log_depth_ghz)
log_depth_ghz = CircuitToMove(
    fixed.LogicalLayoutHeuristic(),
    fixed.LogicalPlacementStrategy(),
    fixed.LogicalMoveScheduler(),
).emit(log_depth_ghz)


arch_spec = logical.get_arch_spec()
frame, _ = AtomInterpreter(log_depth_ghz.dialects, arch_spec=arch_spec).run(
    log_depth_ghz
)
prev_state = None
for stmt in log_depth_ghz.callable_region.walk():
    curr_state = frame.atom_state_map.get(stmt)

    if not isinstance(curr_state, AtomState):
        continue

    ax = plt.gca()

    if prev_state is None:
        prev_state = curr_state
        continue

    word_bus = []
    site_bus = []
    if isinstance(stmt, move.Move):
        for lane in stmt.lanes:
            if lane.move_type == MoveType.WORD:
                word_bus.append(lane.bus_id)
            else:
                site_bus.append(lane.bus_id)

    arch_spec.plot(
        ax,
        show_words=range(len(arch_spec.words)),
        show_word_bus=word_bus,
        show_site_bus=site_bus,
    )

    if isinstance(curr_state, AtomState):
        curr_state.plot(arch_spec, atom_color="red", ax=ax)
        prev_state.plot(arch_spec, atom_color="green", ax=ax)

        prev_state = curr_state

    plt.show()
