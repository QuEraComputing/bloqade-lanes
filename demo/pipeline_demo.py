from bloqade.native.upstream import SquinToNative
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.arch.gemini.logical import SpecializeGemini
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.upstream import CircuitToMove, NativeToCircuit


@squin.kernel(typeinfer=True, fold=True)
def log_depth_ghz():
    size = 10
    q0 = qubit.new()
    squin.h(q0)
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
            squin.broadcast.cx(reg[-num_alloc:], new_qubits)
            reg = reg + new_qubits


log_depth_ghz.print()

log_depth_ghz = SquinToNative().emit(log_depth_ghz)
log_depth_ghz.print()

log_depth_ghz = NativeToCircuit().emit(log_depth_ghz)
log_depth_ghz.print()

log_depth_ghz = CircuitToMove(
    fixed.LogicalLayoutHeuristic(),
    fixed.LogicalPlacementStrategy(),
    fixed.LogicalMoveScheduler(),
).emit(log_depth_ghz)
log_depth_ghz.print()

out = SpecializeGemini().emit(log_depth_ghz)
out.print()
