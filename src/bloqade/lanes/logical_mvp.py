import io

from bloqade.gemini.rewrite.initialize import __RewriteU3ToInitialize
from bloqade.native.upstream import SquinToNative
from bloqade.rewrite.passes import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim
from kirin import ir, passes, rewrite

from bloqade.lanes import visualize
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite import transversal
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.upstream import NativeToPlace, PlaceToMove


def compile_squin_to_move(mt: ir.Method, transversal_rewrite: bool = False):
    """Compile a squin kernel to move dialect.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False
    Returns:
        ir.Method: The compiled move dialect method.
    """

    # Compile to move dialect
    rule = rewrite.Chain(
        rewrite.Walk(
            RewriteNonCliffordToU3(),
        ),
        rewrite.Walk(
            __RewriteU3ToInitialize(),
        ),
    )

    CallGraphPass(mt.dialects, rule)(mt)
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
                transversal.RewriteGetItem(logical.steane7_transversal_map),
                transversal.RewriteLogicalToPhysicalConversion(),
            )
        ).rewrite(mt.code)

    passes.TypeInfer(mt.dialects)(mt)

    mt.verify()
    mt.verify_type()

    return mt


def compile_squin_to_move_and_visualize(
    mt: ir.Method, interactive: bool = True, transversal_rewrite: bool = False
):
    """Compile a squin kernel to moves and visualize the program.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        interactive (bool, optional): Whether to display the visualization interactively. Defaults to True.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
    """
    # Compile to move dialect
    mt = compile_squin_to_move(mt, transversal_rewrite)
    if transversal_rewrite:
        arch_spec = generate_arch(4)
        marker = "o"
    else:
        arch_spec = logical.get_arch_spec()
        marker = "s"

    visualize.debugger(mt, arch_spec, interactive=interactive, atom_marker=marker)


def compile_to_physical_squin_noise_model(
    mt: ir.Method, noise_model: NoiseModelABC | None = None
) -> ir.Method:
    """Compiles a logical squin kernel to a physical squin kernel with noise channels inserted.

    Args:
        mt: The logical squin method to compile.
        noise_model: The noise model to insert during compilation.
    Returns:
        The compiled physical squin method.

    """
    if noise_model is None:
        noise_model = generate_simple_noise_model()

    move_mt = compile_squin_to_move(mt, transversal_rewrite=True)
    transformer = MoveToSquin(
        arch_spec=generate_arch(4),
        logical_initialization=logical.steane7_initialize,
        noise_model=noise_model,
        aggressive_unroll=False,
    )

    return transformer.emit(move_mt)


def compile_to_physical_stim_program(
    mt: ir.Method, noise_model: NoiseModelABC | None = None
) -> str:
    """Compiles a logical squin kernel to a physical stim kernel with noise channels inserted.

    Args:
        mt: The logical squin method to compile.
        noise_model: The noise model to insert during compilation.
    Returns:
        The compiled physical stim program as a string.

    """
    noise_kernel = compile_to_physical_squin_noise_model(mt, noise_model)
    noise_kernel = squin_to_stim(noise_kernel)

    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)

    return buf.getvalue().strip()
