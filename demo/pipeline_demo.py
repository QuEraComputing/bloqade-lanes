from bloqade.gemini.dialects import logical as gemini_logical
from bloqade.native.upstream import SquinToNative
from kirin import rewrite

from bloqade import qubit, squin
from bloqade.lanes.arch.gemini.logical.simulation import rewrite as sim_rewrite
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.upstream import NativeToPlace, PlaceToMove

kernel = squin.kernel.add(gemini_logical.dialect)
kernel.run_pass = squin.kernel.run_pass


@kernel(typeinfer=True, fold=True)
def ghz_optimal():
    size = 10
    qs = qubit.qalloc(size)
    squin.h(qs[0])
    squin.cx(qs[0], qs[1])
    squin.broadcast.cx(qs[:2], qs[2:4])
    squin.cx(qs[3], qs[4])
    squin.broadcast.cx(qs[:5], qs[5:])

    return gemini_logical.terminal_measure(qs)


ghz_optimal.print()

ghz_optimal = SquinToNative().emit(ghz_optimal)

ghz_optimal = NativeToPlace().emit(ghz_optimal)

ghz_optimal = PlaceToMove(
    fixed.LogicalLayoutHeuristic(),
    fixed.LogicalPlacementStrategy(),
    fixed.LogicalMoveScheduler(),
).emit(ghz_optimal)

rewrite.Walk(
    rewrite.Chain(
        sim_rewrite.RewriteLocations(),
        sim_rewrite.RewriteMoves(),
        sim_rewrite.RewriteGetMeasurementResult(),
        sim_rewrite.RewriteLogicalToPhysicalConversion(),
    )
).rewrite(ghz_optimal.code)


ghz_optimal.print()
