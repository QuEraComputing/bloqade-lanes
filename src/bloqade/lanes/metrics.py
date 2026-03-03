from dataclasses import dataclass

from bloqade.analysis.fidelity import FidelityAnalysis, FidelityRange
from kirin import ir
from kirin.validation import ValidationSuite

from bloqade.lanes.analysis.placement.strategy import PlacementStrategyABC
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.layout import ArchSpec, LocationAddress
from bloqade.lanes.logical_mvp import transversal_rewrites
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.upstream import (
    default_merge_heuristic,
    squin_to_move,
)
from bloqade.lanes.validation import address


@dataclass(frozen=True)
class KernelFidelityMetrics:
    """Fidelity metrics computed from a physical noisy SQuin kernel."""

    gate_fidelities: list[float]
    gate_fidelity_product: float


@dataclass(frozen=True)
class KernelMoveMetrics:
    """Move metadata computed from a compiled Move kernel."""

    approx_lane_parallelism: float
    moved_lane_count: int


@dataclass(frozen=True)
class MoveTimeEvent:
    """Per-move event timing details in microseconds.

    ``move_duration_us`` is shared across all lane source addresses.
    """

    event_index: int
    move_type: str
    bus_id: int
    direction: str
    src_addresses: list[LocationAddress]
    move_duration_us: float
    timing_model: str


@dataclass(frozen=True)
class KernelMoveTimeMetrics:
    """Move timing metrics computed from a compiled Move kernel."""

    total_move_time_us: float
    events: list[MoveTimeEvent]
    timing_model: str


def _collapse_range(fidelity: FidelityRange) -> float:
    # Use the conservative lower bound.
    return fidelity.min


def _product_fidelity(fidelities: list[float]) -> float:
    product = 1.0
    for fidelity in fidelities:
        product *= fidelity
    return product


def _count_move_events_and_lanes(move_mt: ir.Method) -> tuple[int, int]:
    move_event_count = 0
    moved_lane_count = 0

    for stmt in move_mt.callable_region.walk():
        if isinstance(stmt, move.Move):
            move_event_count += 1
            moved_lane_count += len(stmt.lanes)

    return move_event_count, moved_lane_count


def _compute_approx_lane_parallelism(
    move_event_count: int, moved_lane_count: int
) -> float:
    if move_event_count == 0:
        return 0.0
    return moved_lane_count / move_event_count


def _compile_kernel_to_noisy_physical_squin(
    mt: ir.Method,
    *,
    placement_strategy: PlacementStrategyABC,
    insert_palindrome_moves: bool,
    merge_heuristic=default_merge_heuristic,
    noise_model: NoiseModelABC | None = None,
) -> ir.Method:
    if noise_model is None:
        noise_model = generate_simple_noise_model()

    move_mt = squin_to_move(
        mt,
        layout_heuristic=fixed.LogicalLayoutHeuristic(),
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
        merge_heuristic=merge_heuristic,
    )
    move_mt = transversal_rewrites(move_mt)
    transformer = MoveToSquin(
        arch_spec=placement_strategy.arch_spec,
        logical_initialization=logical.steane7_initialize,
        noise_model=noise_model,
        aggressive_unroll=False,
    )
    return transformer.emit(move_mt)


def _analyze_move_time_from_move_ir(
    move_mt: ir.Method,
    *,
    arch_spec: ArchSpec,
    flair_amplitude_delta: float,
) -> KernelMoveTimeMetrics:
    timing_model = "arch_spec_get_lane_duration_us"
    events: list[MoveTimeEvent] = []
    for event_index, stmt in enumerate(move_mt.callable_region.walk()):
        if not isinstance(stmt, move.Move):
            continue

        lane_durations_us: list[float] = []

        for lane in stmt.lanes:
            lane_duration_us = arch_spec.get_lane_duration_us(
                lane,
                amplitude_delta=flair_amplitude_delta,
            )

            lane_durations_us.append(lane_duration_us)

        if len(lane_durations_us) == 0:
            continue

        move_duration_us = lane_durations_us[0]
        if any(duration != move_duration_us for duration in lane_durations_us[1:]):
            raise ValueError(
                "All lanes in a move event must have the same duration; "
                f"got {lane_durations_us}"
            )
        rep_lane = stmt.lanes[0]
        if any(
            lane.move_type != rep_lane.move_type
            or lane.bus_id != rep_lane.bus_id
            or lane.direction != rep_lane.direction
            for lane in stmt.lanes[1:]
        ):
            raise ValueError(
                "All lanes in a move event must share move_type, bus_id, and direction"
            )

        events.append(
            MoveTimeEvent(
                event_index=event_index,
                move_type=rep_lane.move_type.name,
                bus_id=rep_lane.bus_id,
                direction=rep_lane.direction.name,
                src_addresses=[lane.src_site() for lane in stmt.lanes],
                move_duration_us=move_duration_us,
                timing_model=timing_model,
            )
        )

    total_move_time_us = sum(event.move_duration_us for event in events)
    return KernelMoveTimeMetrics(
        total_move_time_us=total_move_time_us,
        events=events,
        timing_model=timing_model,
    )


def analyze_kernel_fidelity_with_strategy(
    mt: ir.Method,
    *,
    placement_strategy: PlacementStrategyABC,
    insert_palindrome_moves: bool,
    merge_heuristic=default_merge_heuristic,
    noise_model: NoiseModelABC | None = None,
) -> KernelFidelityMetrics:
    """
    Analyze approximate fidelity for a logical SQuin kernel with explicit strategy control.

    The kernel is compiled through the lanes pipeline:
    logical SQuin -> move -> physical noisy SQuin. The resulting physical kernel
    is then passed to ``FidelityAnalysis`` from ``bloqade-circuit``.

    We intentionally report only pauli gate (XYZ) fidelities for now. With the
    current GeminiNoiseModelABC defaults, all loss probabilities are 0. The reported gate
    fidelity includes non-loss errors from moves, idling, and CZ-unpaired noise.
    If upstream returns a fidelity range, this uses the conservative lower.
    """
    physical_squin = _compile_kernel_to_noisy_physical_squin(
        mt,
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
        merge_heuristic=merge_heuristic,
        noise_model=noise_model,
    )
    analysis = FidelityAnalysis(physical_squin.dialects)
    analysis.run(physical_squin)
    gate_fidelities = [_collapse_range(fid) for fid in analysis.gate_fidelities]
    return KernelFidelityMetrics(
        gate_fidelities=gate_fidelities,
        gate_fidelity_product=_product_fidelity(gate_fidelities),
    )


def analyze_kernel_moves_with_strategy(
    mt: ir.Method,
    *,
    placement_strategy: PlacementStrategyABC,
    insert_palindrome_moves: bool,
    merge_heuristic=default_merge_heuristic,
) -> KernelMoveMetrics:
    """
    Analyze move-count metadata with explicit move-compilation strategy control.

    The kernel is compiled to Move IR and then scanned for ``move.Move``
    statements. We report:
    - ``approx_lane_parallelism`` = moved_lane_count / move_event_count
    - ``moved_lane_count`` = total lanes carried by all move events
    """
    move_mt = squin_to_move(
        mt,
        layout_heuristic=fixed.LogicalLayoutHeuristic(),
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
        merge_heuristic=merge_heuristic,
    )
    validator = ValidationSuite(
        [
            address.assign_arch_spec(
                address.Validation, arch_spec_to_assign=placement_strategy.arch_spec
            )
        ]
    )
    validator.validate(move_mt).raise_if_invalid()

    move_event_count, moved_lane_count = _count_move_events_and_lanes(move_mt)
    return KernelMoveMetrics(
        approx_lane_parallelism=_compute_approx_lane_parallelism(
            move_event_count, moved_lane_count
        ),
        moved_lane_count=moved_lane_count,
    )


def analyze_kernel_move_time_with_strategy(
    mt: ir.Method,
    *,
    placement_strategy: PlacementStrategyABC,
    insert_palindrome_moves: bool,
    merge_heuristic=default_merge_heuristic,
    flair_amplitude_delta: float = 1.0,
) -> KernelMoveTimeMetrics:
    """
    Analyze move-time metadata with explicit move-compilation strategy control.

    The kernel is compiled to Move IR and each ``move.Move`` statement is
    treated as one move event. Per-lane duration is delegated to
    ``ArchSpec.get_lane_duration_us(lane, amplitude_delta=...)`` from the
    architecture owned by ``placement_strategy``. Events record one shared
    duration and the source location addresses moved during the event.
    """
    move_mt = squin_to_move(
        mt,
        layout_heuristic=fixed.LogicalLayoutHeuristic(),
        placement_strategy=placement_strategy,
        insert_palindrome_moves=insert_palindrome_moves,
        merge_heuristic=merge_heuristic,
    )
    return _analyze_move_time_from_move_ir(
        move_mt,
        arch_spec=placement_strategy.arch_spec,
        flair_amplitude_delta=flair_amplitude_delta,
    )
