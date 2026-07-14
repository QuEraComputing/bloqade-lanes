import io
from typing import Any, Callable

from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim
from kirin import ir
from kirin.validation import ValidationSuite

from bloqade.gemini.logical.validation.clifford.analysis import GeminiLogicalValidation
from bloqade.gemini.logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.gemini.measurement_annotations import (
    _find_qubit_ssas as _find_qubit_ssas,
    _find_return_stmt as _find_return_stmt,
    append_measurements_and_annotations,
)
from bloqade.lanes import visualize
from bloqade.lanes.analysis import atom, placement
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini import logical, physical
from bloqade.lanes.cudaq_integration import cudaq_to_squin, is_cudaq_kernel
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.lanes.pipeline import LogicalPipeline
from bloqade.lanes.pipeline.logical import transversal_rewrites
from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs
from bloqade.lanes.transform import MoveToSquinLogical

__all__ = [
    "run_squin_kernel_validation",
    "compile_squin_to_move",
    "compile_squin_to_move_and_visualize",
    "compile_to_physical_squin_noise_model",
    "compile_to_stim_program",
    "compile_task",
    "transversal_rewrites",
    "append_measurements_and_annotations",
]


def run_squin_kernel_validation(mt: ir.Method):
    """
    Run validation checks on a Squin kernel method.

    Args:
        mt (ir.Method): The Squin kernel method to validate.

    Returns:
        ValidationResult: A validation result object containing the
            validation errors, if they exist

    Note: To trigger an error run `run_squin_kernel_validation(mt).raise_if_invalid()`.

    """
    validator = ValidationSuite(
        [
            GeminiLogicalValidation,
            GeminiTerminalMeasurementValidation,
            FlatKernelNoCloningValidation,
        ]
    )
    return validator.validate(mt)


def compile_squin_to_move(
    mt: ir.Method,
    transversal_rewrite: bool = False,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: placement.PlacementStrategyABC | None = None,
):
    """Compile a squin kernel to the move dialect.

    Args:
        mt (ir.Method): The squin kernel to compile.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.
        layout_heuristic (LayoutHeuristicABC | None, optional): Layout heuristic for atom
            placement. Defaults to ``None`` (uses ``LogicalLayoutHeuristic``).
        placement_strategy (PlacementStrategyABC | None, optional): Placement strategy.
            Defaults to ``None`` (uses ``PalindromePlacementStrategy`` wrapping
            ``LogicalPlacementStrategyNoHome``). Pass a bare strategy without
            ``PalindromePlacementStrategy`` to disable palindrome return moves.

    Returns:
        ir.Method: The compiled move dialect method.

    """

    return LogicalPipeline(
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        transversal_rewrite=transversal_rewrite,
    ).emit(mt, no_raise=no_raise)


def compile_squin_to_move_and_visualize(
    mt: ir.Method,
    interactive: bool = True,
    transversal_rewrite: bool = False,
    animated: bool = False,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
):
    """Compile a squin kernel to moves and visualize the program.

    Args:
        mt (ir.Method): The squin kernel to compile.
        interactive (bool, optional): Whether to display the visualization interactively. Defaults to True.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
        animated (bool, optional): Whether to use animated visualization for displaying moves. Defaults to False.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.
        layout_heuristic (LayoutHeuristicABC | None, optional): Layout heuristic for atom
            placement. Defaults to ``None`` (uses ``LogicalLayoutHeuristic``).

    """
    # Compile to move dialect
    mt = compile_squin_to_move(
        mt,
        transversal_rewrite,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
    )
    if transversal_rewrite:
        arch_spec = physical.get_arch_spec()
        marker = "o"
    else:
        arch_spec = logical.get_arch_spec()
        marker = "s"

    if animated:
        visualize.animated_debugger(
            mt, arch_spec, interactive=interactive, atom_marker=marker
        )
    else:
        visualize.debugger(mt, arch_spec, interactive=interactive, atom_marker=marker)


def compile_to_physical_squin_noise_model(
    mt: ir.Method,
    noise_model: LogicalNoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
) -> ir.Method:
    """Compile a logical squin kernel to a physical squin kernel with noise channels inserted.

    Args:
        mt (ir.Method): The logical squin method to compile.
        noise_model (LogicalNoiseModelABC | None, optional): The logical noise model to insert
            during compilation. Defaults to ``None`` (uses :func:`generate_logical_noise_model`).
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.
        layout_heuristic (LayoutHeuristicABC | None, optional): Layout heuristic for atom
            placement. Defaults to ``None`` (uses ``LogicalLayoutHeuristic``).

    Returns:
        ir.Method: The compiled physical squin method with noise channels.

    """
    if noise_model is None:
        noise_model = generate_logical_noise_model()

    move_mt = compile_squin_to_move(
        mt,
        transversal_rewrite=True,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
    )
    transformer = MoveToSquinLogical(
        arch_spec=physical.get_arch_spec(),
        noise_model=noise_model,
        add_noise=True,
        aggressive_unroll=False,
    )
    return transformer.emit(move_mt, no_raise=no_raise)


def compile_to_stim_program(
    mt: ir.Method,
    noise_model: LogicalNoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
) -> str:
    """Compile a logical squin kernel to a Stim program string with noise channels inserted.

    Args:
        mt (ir.Method): The logical squin method to compile.
        noise_model (NoiseModelABC | None, optional): The noise model to insert during
            compilation. Defaults to ``None`` (uses :func:`generate_simple_noise_model`).
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.
        layout_heuristic (LayoutHeuristicABC | None, optional): Layout heuristic for atom
            placement. Defaults to ``None`` (uses ``LogicalLayoutHeuristic``).

    Returns:
        str: The compiled Stim program as a string.

    """
    noise_kernel = compile_to_physical_squin_noise_model(
        mt,
        noise_model,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
    )
    RemoveReturn().rewrite(noise_kernel.code)
    noise_kernel = squin_to_stim(noise_kernel)
    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)
    return buf.getvalue().strip()


def compile_task(
    logical_kernel: ir.Method | Callable[..., Any],
    m2dets: list[list[int]] | None = None,
    m2obs: list[list[int]] | None = None,
):
    """Compile a logical kernel into physical move artifacts.

    Handles CUDAQ kernel detection/conversion, squin kernel validation,
    squin-to-move compilation, architecture spec generation, and
    post-processing extraction.

    Args:
        logical_kernel: A squin ``ir.Method`` or a CUDA-Q kernel to compile.
        m2dets: Binary measurement-to-detector matrix. For CUDA-Q kernels,
            defaults to Steane [[7,1,3]] detectors if ``None``.
        m2obs: Binary measurement-to-observable matrix. For CUDA-Q kernels,
            defaults to Steane [[7,1,3]] observables if ``None``.

    Returns:
        A tuple of ``(logical_squin_kernel, physical_arch_spec,
        physical_move_kernel, post_processing)``.

    """
    if is_cudaq_kernel(logical_kernel):
        logical_squin_kernel: ir.Method = cudaq_to_squin(logical_kernel)

        if m2dets is None and m2obs is None:
            num_qubits = len(_find_qubit_ssas(logical_squin_kernel))
            m2dets = steane7_m2dets(num_qubits)
            m2obs = steane7_m2obs(num_qubits)

        append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
    elif isinstance(logical_kernel, ir.Method):
        logical_squin_kernel = logical_kernel
        if m2dets is not None or m2obs is not None:
            append_measurements_and_annotations(logical_squin_kernel, m2dets, m2obs)
    else:
        raise ValueError(f"Unknown kernel type {type(logical_kernel)}")

    run_squin_kernel_validation(logical_squin_kernel).raise_if_invalid()

    physical_arch_spec = physical.get_arch_spec()
    physical_move_kernel = compile_squin_to_move(
        logical_squin_kernel, transversal_rewrite=True
    )
    post_processing = atom.AtomInterpreter(
        physical_move_kernel.dialects, arch_spec=physical_arch_spec
    ).get_post_processing(physical_move_kernel)

    return (
        logical_squin_kernel,
        physical_arch_spec,
        physical_move_kernel,
        post_processing,
    )
