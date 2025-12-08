from bloqade.native.upstream import SquinToNative
from kirin import ir, rewrite
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes import visualize
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.arch.gemini.logical.simulation import rewrite as sim_rewrite
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.upstream import NativeToPlace, PlaceToMove


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


@squin.kernel(typeinfer=True, fold=True)
def ghz_optimal():
    size = 10
    qs = qubit.qalloc(size)
    squin.h(qs[0])
    squin.cx(qs[0], qs[1])
    squin.broadcast.cx(qs[:2], qs[2:4])
    squin.cx(qs[3], qs[4])
    squin.broadcast.cx(qs[:5], qs[5:])


def compile(mt: ir.Method, transversal: bool = False):
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
            rewrite.Chain(sim_rewrite.RewriteLocations(), sim_rewrite.RewriteMoves())
        ).rewrite(mt.code)
    return mt


def compile_and_visualize(mt: ir.Method, interactive=True, transversal: bool = False):
    # Compile to move dialect
    mt = compile(mt, transversal=transversal)
    if transversal:
        arch_spec = generate_arch(4)
        marker = "o"
    else:
        arch_spec = logical.get_arch_spec()
        marker = "s"

    visualize.debugger(mt, arch_spec, interactive=interactive, atom_marker=marker)


compile_and_visualize(log_depth_ghz, transversal=False)
compile_and_visualize(ghz_optimal, transversal=False)
compile_and_visualize(ghz_optimal, transversal=True)
