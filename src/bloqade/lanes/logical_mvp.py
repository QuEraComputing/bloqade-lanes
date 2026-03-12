import io
from typing import Literal

from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.gemini.analysis.logical_validation import GeminiLogicalValidation
from bloqade.gemini.analysis.measurement_validation import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim
from kirin import ir, rewrite
from kirin.validation import ValidationSuite

from bloqade.lanes import visualize
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.heuristics.logical_placement import (
    LogicalPlacementStrategy,
    LogicalPlacementStrategyNoHome,
)
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicFixed,
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import PhysicalGreedyPlacementStrategy
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite import transversal
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.upstream import squin_to_move


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


def transversal_rewrites(mt: ir.Method):
    """Apply transversal rewrite rules to a squin method.

    Args:
        mt (ir.Method): rewrite the method in place.

    Returns:
        ir.Method: The rewritten method.

    """

    rewrite.Walk(
        rewrite.Chain(
            transversal.RewriteLocations(logical.steane7_transversal_map),
            transversal.RewriteLogicalInitialize(logical.steane7_transversal_map),
            transversal.RewriteMoves(logical.steane7_transversal_map),
            transversal.RewriteGetItem(logical.steane7_transversal_map),
            transversal.RewriteLogicalToPhysicalConversion(),
        )
    ).rewrite(mt.code)

    return mt


def compile_squin_to_move(
    mt: ir.Method,
    transversal_rewrite: bool = False,
    no_raise: bool = True,
    placement_mode: Literal["logical", "physical"] = "logical",
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_palindrome_moves: bool = True,
):
    """
    Compile a squin kernel to move dialect.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.

    Returns:
        ir.Method: The compiled move dialect method.
    """

    def _layout_mode(
        layout_h: LayoutHeuristicABC,
    ) -> Literal["logical", "physical"] | None:
        if isinstance(
            layout_h,
            (
                PhysicalLayoutHeuristicFixed,
                PhysicalLayoutHeuristicGraphPartitionCenterOut,
            ),
        ):
            return "physical"
        if isinstance(layout_h, logical_layout.LogicalLayoutHeuristic):
            return "logical"
        return None

    def _strategy_mode(
        strategy: PlacementStrategyABC,
    ) -> Literal["logical", "physical"] | None:
        if isinstance(strategy, PhysicalGreedyPlacementStrategy):
            return "physical"
        if isinstance(
            strategy, (LogicalPlacementStrategy, LogicalPlacementStrategyNoHome)
        ):
            return "logical"
        return None

    if placement_mode not in ("logical", "physical"):
        raise ValueError(
            f"Unsupported placement_mode={placement_mode!r}; expected 'logical' or 'physical'."
        )
    physical_arch_spec = None
    if placement_mode == "physical" and (
        layout_heuristic is None or placement_strategy is None
    ):
        physical_arch_spec = get_physical_arch_spec()
    if layout_heuristic is None:
        if placement_mode == "logical":
            layout_heuristic = logical_layout.LogicalLayoutHeuristic()
        else:
            assert physical_arch_spec is not None
            layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(
                arch_spec=physical_arch_spec
            )
    if placement_strategy is None:
        if placement_mode == "logical":
            placement_strategy = LogicalPlacementStrategyNoHome()
        else:
            assert physical_arch_spec is not None
            placement_strategy = PhysicalGreedyPlacementStrategy(
                arch_spec=physical_arch_spec
            )

    layout_mode = _layout_mode(layout_heuristic)
    strategy_mode = _strategy_mode(placement_strategy)
    if layout_mode is not None and layout_mode != placement_mode:
        raise ValueError(
            "layout_heuristic is incompatible with placement_mode="
            f"{placement_mode!r}; inferred mode is {layout_mode!r}."
        )
    if strategy_mode is not None and strategy_mode != placement_mode:
        raise ValueError(
            "placement_strategy is incompatible with placement_mode="
            f"{placement_mode!r}; inferred mode is {strategy_mode!r}."
        )
    mt = squin_to_move(
        mt,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
        no_raise=no_raise,
    )
    if transversal_rewrite:
        mt = transversal_rewrites(mt)

    return mt


def compile_squin_to_move_and_visualize(
    mt: ir.Method,
    interactive: bool = True,
    transversal_rewrite: bool = False,
    animated: bool = False,
    no_raise: bool = True,
    placement_mode: Literal["logical", "physical"] = "logical",
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_palindrome_moves: bool = True,
):
    """
    Compile a squin kernel to moves and visualize the program.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        interactive (bool, optional): Whether to display the visualization interactively. Defaults to True.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
        animated (bool, optional): Whether to use animated visualization for displaying moves. Defaults to False.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.
    """
    # Compile to move dialect
    mt = compile_squin_to_move(
        mt,
        transversal_rewrite,
        no_raise=no_raise,
        placement_mode=placement_mode,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
    )
    use_physical_visualization = (
        transversal_rewrite
        or placement_mode == "physical"
        or isinstance(
            layout_heuristic,
            (
                PhysicalLayoutHeuristicFixed,
                PhysicalLayoutHeuristicGraphPartitionCenterOut,
            ),
        )
        or isinstance(placement_strategy, PhysicalGreedyPlacementStrategy)
    )
    if use_physical_visualization:
        arch_spec = get_physical_arch_spec()
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
    noise_model: NoiseModelABC | None = None,
    no_raise: bool = True,
    arch_spec=None,
    placement_mode: Literal["logical", "physical"] = "physical",
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_palindrome_moves: bool = True,
) -> ir.Method:
    """
    Compiles a logical squin kernel to a physical squin kernel with noise channels inserted.

    Args:
        mt (ir.Method): The logical squin method to compile.
        noise_model (NoiseModelABC, optional): The noise model to insert during compilation. Defaults to None.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.

    Returns:
        ir.Method: The compiled physical squin method.
    """
    if noise_model is None:
        noise_model = generate_simple_noise_model()
    if arch_spec is None:
        arch_spec = get_physical_arch_spec()
    use_physical_defaults = (
        placement_mode == "physical" and move.dialect not in mt.dialects
    )
    if use_physical_defaults:
        if layout_heuristic is None:
            layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(
                arch_spec=arch_spec
            )
        if placement_strategy is None:
            placement_strategy = PhysicalGreedyPlacementStrategy(arch_spec=arch_spec)

    move_mt = compile_squin_to_move(
        mt,
        transversal_rewrite=True,
        no_raise=no_raise,
        placement_mode=placement_mode,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
    )
    transformer = MoveToSquin(
        arch_spec=arch_spec,
        logical_initialization=logical.steane7_initialize,
        noise_model=noise_model,
        aggressive_unroll=False,
    )

    return transformer.emit(move_mt, no_raise=no_raise)


def compile_to_physical_stim_program(
    mt: ir.Method,
    noise_model: NoiseModelABC | None = None,
    no_raise: bool = True,
    arch_spec=None,
    placement_mode: Literal["logical", "physical"] = "physical",
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_palindrome_moves: bool = True,
) -> str:
    """
    Compiles a logical squin kernel to a physical stim kernel with noise channels inserted.

    Args:
        mt (ir.Method): The logical squin method to compile.
        noise_model (NoiseModelABC, optional): The noise model to insert during compilation. Defaults to None.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.

    Returns:
        str: The compiled physical stim program as a string.
    """
    noise_kernel = compile_to_physical_squin_noise_model(
        mt,
        noise_model,
        no_raise=no_raise,
        arch_spec=arch_spec,
        placement_mode=placement_mode,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
    )
    RemoveReturn().rewrite(noise_kernel.code)
    noise_kernel = squin_to_stim(noise_kernel)
    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)

    return buf.getvalue().strip()
