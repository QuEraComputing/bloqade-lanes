from bloqade.native.upstream import SquinToNative
from kirin import ir, rewrite

from bloqade.lanes import visualize
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.rewrite import transversal
from bloqade.lanes.upstream import NativeToPlace, PlaceToMove


def compile_squin(mt: ir.Method, transversal_rewrite: bool = False):
    # Compile to move dialect

    mt = SquinToNative().emit(mt)
    mt = NativeToPlace().emit(mt)
    mt = PlaceToMove(
        fixed.LogicalLayoutHeuristic(),
        fixed.LogicalPlacementStrategy(),
        fixed.LogicalMoveScheduler(),
    ).emit(mt)

    if transversal_rewrite:
        rewrite.Walk(
            rewrite.Chain(
                transversal.RewriteLocations(logical.steane7_transversal_map),
                transversal.RewriteLogicalInitialize(logical.steane7_transversal_map),
                transversal.RewriteMoves(logical.steane7_transversal_map),
                transversal.RewriteGetMeasurementResult(
                    logical.steane7_transversal_map
                ),
                transversal.RewriteLogicalToPhysicalConversion(),
            )
        ).rewrite(mt.code)
    return mt


def compile_and_visualize(
    mt: ir.Method, interactive: bool = True, transversal_rewrite: bool = False
):
    # Compile to move dialect
    mt = compile_squin(mt, transversal_rewrite)
    if transversal_rewrite:
        arch_spec = generate_arch(4)
        marker = "o"
    else:
        arch_spec = logical.get_arch_spec()
        marker = "s"

    visualize.debugger(mt, arch_spec, interactive=interactive, atom_marker=marker)
