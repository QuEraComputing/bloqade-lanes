import io
from collections.abc import Sequence

from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim
from kirin import ir

from bloqade.lanes import visualize
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.pipeline import PhysicalPipeline
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.transform import MoveToSquinPhysical

__all__ = [
    "compile_squin_to_move",
    "compile_squin_to_move_and_visualize",
    "compile_squin_to_move_best",
    "compile_to_physical_squin_noise_model",
    "compile_to_stim_program",
]


def compile_squin_to_move(
    mt: ir.Method,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_return_moves: bool = True,
) -> ir.Method:
    """Compile a physical squin kernel to the move dialect."""
    return PhysicalPipeline(
        arch_spec=get_physical_arch_spec(),
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_return_moves=insert_return_moves,
    ).emit(mt, no_raise=no_raise)


def _count_move_events(mt: ir.Method) -> int:
    """Count move.Move statements in a compiled move-dialect method.

    One move.Move statement corresponds to one AOD shot; the total is the
    comparison metric used by :func:`compile_squin_to_move_best`.
    """
    return sum(1 for stmt in mt.callable_region.walk() if isinstance(stmt, move.Move))


def compile_squin_to_move_best(
    mt: ir.Method,
    strategies: Sequence[tuple[str, PlacementStrategyABC]],
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
    insert_return_moves: bool = True,
) -> tuple[ir.Method, str]:
    """Compile with each ``(label, strategy)`` and return the one producing
    the fewest :class:`move.Move` events.

    Ties are resolved by earliest-in-list (caller-controlled preference).
    The returned ``label`` names the winning strategy.

    The three intra-stage heuristics
    (:class:`DefaultTargetGenerator`,
    :class:`CongestionAwareTargetGenerator`,
    :class:`AODClusterTargetGenerator`) each win on different circuit
    shapes. Racing all three and keeping the best covers the per-circuit
    variation without designing a meta-heuristic.
    """
    if not strategies:
        raise ValueError("compile_squin_to_move_best requires at least one strategy")

    best_mt: ir.Method | None = None
    best_events = -1
    best_label = ""
    for label, strategy in strategies:
        compiled = compile_squin_to_move(
            mt,
            no_raise=no_raise,
            layout_heuristic=layout_heuristic,
            placement_strategy=strategy,
            insert_return_moves=insert_return_moves,
        )
        events = _count_move_events(compiled)
        # strict-less keeps the earliest on ties.
        if best_mt is None or events < best_events:
            best_mt = compiled
            best_events = events
            best_label = label
    assert best_mt is not None
    return best_mt, best_label


def compile_squin_to_move_and_visualize(
    mt: ir.Method,
    interactive: bool = True,
    animated: bool = False,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_return_moves: bool = True,
) -> None:
    """Compile a physical squin kernel to moves and visualize the program."""
    mt = compile_squin_to_move(
        mt,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_return_moves=insert_return_moves,
    )
    arch_spec = get_physical_arch_spec()
    marker = "o"

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
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_return_moves: bool = True,
) -> ir.Method:
    """Compile a physical squin kernel to physical squin with inserted noise channels."""
    if noise_model is None:
        noise_model = generate_simple_noise_model()
    if arch_spec is None:
        arch_spec = get_physical_arch_spec()

    move_mt = compile_squin_to_move(
        mt,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_return_moves=insert_return_moves,
    )
    transformer = MoveToSquinPhysical(
        arch_spec=arch_spec,
        noise_model=noise_model,
        aggressive_unroll=False,
    )

    return transformer.emit(move_mt, no_raise=no_raise)


def compile_to_stim_program(
    mt: ir.Method,
    noise_model: NoiseModelABC | None = None,
    no_raise: bool = True,
    arch_spec=None,
    layout_heuristic: LayoutHeuristicABC | None = None,
    placement_strategy: PlacementStrategyABC | None = None,
    insert_return_moves: bool = True,
) -> str:
    """Compile a physical squin kernel to a Stim program string."""
    noise_kernel = compile_to_physical_squin_noise_model(
        mt,
        noise_model=noise_model,
        no_raise=no_raise,
        arch_spec=arch_spec,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        insert_return_moves=insert_return_moves,
    )
    RemoveReturn().rewrite(noise_kernel.code)
    noise_kernel = squin_to_stim(noise_kernel)
    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)
    return buf.getvalue().strip()
