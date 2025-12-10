from bloqade.native.upstream import SquinToNative
from kirin import ir, rewrite

from bloqade.lanes import visualize
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.arch.gemini.logical.simulation import rewrite as sim_rewrite
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.upstream import NativeToPlace, PlaceToMove


def compile_squin(mt: ir.Method, transversal: bool = False):
    # Compile to move dialect

    mt = SquinToNative().emit(mt)
    mt = NativeToPlace().emit(mt)
    mt = PlaceToMove(
        fixed.LogicalLayoutHeuristic(),
        fixed.LogicalPlacementStrategy(),
        fixed.LogicalMoveScheduler(),
    ).emit(mt)

    if transversal:
        rewrite.Walk(
            rewrite.Chain(
                sim_rewrite.RewriteLocations(),
                sim_rewrite.RewriteMoves(),
                sim_rewrite.RewriteGetMeasurementResult(),
                sim_rewrite.RewriteLogicalToPhysicalConversion(),
            )
        ).rewrite(mt.code)
    return mt


def compile_and_visualize(
    mt: ir.Method, interactive: bool = True, transversal: bool = False
):
    # Compile to move dialect
    mt = compile_squin(mt, transversal=transversal)
    if transversal:
        arch_spec = generate_arch(4)
        marker = "o"
    else:
        arch_spec = logical.get_arch_spec()
        marker = "s"

    visualize.debugger(mt, arch_spec, interactive=interactive, atom_marker=marker)
