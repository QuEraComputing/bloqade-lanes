from bloqade.gemini.dialects import logical as gemini_logical
from bloqade.native.upstream import SquinToNative
from kirin import rewrite
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.arch.gemini.logical.simulation import rewrite as sim_rewrite
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.upstream import NativeToPlace, PlaceToMove

kernel = squin.kernel.add(gemini_logical.dialect)
kernel.run_pass = squin.kernel.run_pass


@kernel(typeinfer=True, fold=True)
def main():
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

    return gemini_logical.terminal_measure(reg)


main.print()

main = SquinToNative().emit(main)
main = NativeToPlace().emit(main)
main.print()

main = PlaceToMove(
    fixed.LogicalLayoutHeuristic(),
    fixed.LogicalPlacementStrategy(),
    fixed.LogicalMoveScheduler(),
).emit(main)
# transversal rewrite to convert logical to physical addresses
rewrite.Walk(
    rewrite.Chain(
        sim_rewrite.RewriteLocations(),
        sim_rewrite.RewriteMoves(),
        sim_rewrite.RewriteGetMeasurementResult(),
        sim_rewrite.RewriteLogicalToPhysicalConversion(),
    )
).rewrite(main.code)


main.print()
