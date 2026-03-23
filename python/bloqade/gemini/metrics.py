from dataclasses import dataclass, field
from typing import Any

from bloqade.analysis.fidelity import FidelityAnalysis, FidelityRange
from kirin import ir

from bloqade.gemini.arch import logical
from bloqade.gemini.compile import transversal_rewrites
from bloqade.gemini.noise_model import generate_simple_noise_model
from bloqade.lanes.analysis.placement.strategy import PlacementStrategyABC
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.layout.move_metric import MoveMetricCalculator
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.upstream import (
    default_merge_heuristic,
    squin_to_move,
)


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
    """Per-move event timing details in microseconds."""

    event_index: int
    lane_count: int
    move_type: str
    bus_id: int
    direction: str
    lane_durations_us: list[float]
    event_duration_us: float
    segment_distances_um: list[float]
    segment_durations_us: list[float]
    pick_time_us: float
    drop_time_us: float
    timing_model: str


@dataclass(frozen=True)
class KernelMoveTimeMetrics:
    """Move timing metrics computed from a compiled Move kernel."""

    total_move_time_us: float
    events: list[MoveTimeEvent]
    timing_model: str


@dataclass
class Metrics:
    """Unified metrics computation for the lanes pipeline.

    Owns kernel-level analysis methods and delegates all move-metric
    computation (lane durations, costs, distances) to a
    ``MoveMetricCalculator`` instance.
    """

    arch_spec: Any  # ArchSpec — use Any to avoid circular import
    noise_model: NoiseModelABC | None = None
    move_calc: MoveMetricCalculator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.move_calc = MoveMetricCalculator(arch_spec=self.arch_spec)

    # --- Private helpers ---

    def _compile_to_noisy_physical_squin(
        self,
        mt: ir.Method,
        *,
        placement_strategy: PlacementStrategyABC,
        insert_return_moves: bool,
        merge_heuristic=default_merge_heuristic,
    ) -> ir.Method:
        noise_model = self.noise_model
        if noise_model is None:
            noise_model = generate_simple_noise_model()

        move_mt = squin_to_move(
            mt,
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(
                arch_spec=logical.get_arch_spec()
            ),
            placement_strategy=placement_strategy,
            insert_return_moves=insert_return_moves,
            merge_heuristic=merge_heuristic,
        )
        move_mt = transversal_rewrites(move_mt)
        transformer = MoveToSquin(
            arch_spec=self.arch_spec,
            logical_initialization=logical.steane7_initialize,
            noise_model=noise_model,
            aggressive_unroll=False,
        )
        return transformer.emit(move_mt)

    # --- High-level analysis methods ---

    def analyze_fidelity(
        self,
        mt: ir.Method,
        *,
        placement_strategy: PlacementStrategyABC,
        insert_return_moves: bool,
        merge_heuristic=default_merge_heuristic,
    ) -> KernelFidelityMetrics:
        physical_squin = self._compile_to_noisy_physical_squin(
            mt,
            placement_strategy=placement_strategy,
            insert_return_moves=insert_return_moves,
            merge_heuristic=merge_heuristic,
        )
        analysis = FidelityAnalysis(physical_squin.dialects)
        analysis.run(physical_squin)
        gate_fidelities = [_collapse_range(fid) for fid in analysis.gate_fidelities]
        return KernelFidelityMetrics(
            gate_fidelities=gate_fidelities,
            gate_fidelity_product=_product_fidelity(gate_fidelities),
        )

    def analyze_moves(
        self,
        mt: ir.Method,
        *,
        placement_strategy: PlacementStrategyABC,
        insert_return_moves: bool,
        merge_heuristic=default_merge_heuristic,
    ) -> KernelMoveMetrics:
        move_mt = squin_to_move(
            mt,
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(
                arch_spec=logical.get_arch_spec()
            ),
            placement_strategy=placement_strategy,
            insert_return_moves=insert_return_moves,
            merge_heuristic=merge_heuristic,
        )
        move_event_count, moved_lane_count = _count_move_events_and_lanes(move_mt)
        return KernelMoveMetrics(
            approx_lane_parallelism=_compute_approx_lane_parallelism(
                move_event_count, moved_lane_count
            ),
            moved_lane_count=moved_lane_count,
        )

    def analyze_move_time(
        self,
        mt: ir.Method,
        *,
        placement_strategy: PlacementStrategyABC,
        insert_return_moves: bool,
        merge_heuristic=default_merge_heuristic,
        flair_amplitude_delta: float = 1.0,
    ) -> KernelMoveTimeMetrics:
        move_mt = squin_to_move(
            mt,
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(
                arch_spec=logical.get_arch_spec()
            ),
            placement_strategy=placement_strategy,
            insert_return_moves=insert_return_moves,
            merge_heuristic=merge_heuristic,
        )
        return self.analyze_move_time_from_move_ir(
            move_mt,
            flair_amplitude_delta=flair_amplitude_delta,
        )

    # --- Low-level analysis methods ---

    def analyze_move_time_from_move_ir(
        self,
        move_mt: ir.Method,
        flair_amplitude_delta: float = 1.0,
    ) -> KernelMoveTimeMetrics:
        mc = self.move_calc
        timing_model = "flair_extracted_const_jerk"
        events: list[MoveTimeEvent] = []
        for event_index, stmt in enumerate(move_mt.callable_region.walk()):
            if not isinstance(stmt, move.Move):
                continue

            lane_durations_us: list[float] = []
            lane_segment_distances_um: list[list[float]] = []
            lane_segment_durations_us: list[list[float]] = []
            lane_pick_times_us: list[float] = []
            lane_drop_times_us: list[float] = []

            for lane in stmt.lanes:
                path = self.arch_spec.get_path(lane)
                segment_distances_um = list(mc.path_segment_distances_um(path))
                segment_durations_us = [
                    mc._const_jerk_min_duration_us(d) for d in segment_distances_um
                ]
                normalized_amp = abs(float(flair_amplitude_delta))
                ramp_time_us = normalized_amp / mc._FLAIR_MAX_RAMP_US
                lane_duration_us = (
                    ramp_time_us + sum(segment_durations_us) + ramp_time_us
                )

                lane_durations_us.append(lane_duration_us)
                lane_segment_distances_um.append(segment_distances_um)
                lane_segment_durations_us.append(segment_durations_us)
                lane_pick_times_us.append(ramp_time_us)
                lane_drop_times_us.append(ramp_time_us)

            event_duration_us = _compute_event_duration_us(lane_durations_us)
            if len(lane_durations_us) == 0:
                continue

            rep_index = max(
                range(len(lane_durations_us)), key=lane_durations_us.__getitem__
            )
            rep_lane = stmt.lanes[rep_index]
            events.append(
                MoveTimeEvent(
                    event_index=event_index,
                    lane_count=len(stmt.lanes),
                    move_type=rep_lane.move_type.name,
                    bus_id=rep_lane.bus_id,
                    direction=rep_lane.direction.name,
                    lane_durations_us=lane_durations_us,
                    event_duration_us=event_duration_us,
                    segment_distances_um=lane_segment_distances_um[rep_index],
                    segment_durations_us=lane_segment_durations_us[rep_index],
                    pick_time_us=lane_pick_times_us[rep_index],
                    drop_time_us=lane_drop_times_us[rep_index],
                    timing_model=timing_model,
                )
            )

        total_move_time_us = sum(event.event_duration_us for event in events)
        return KernelMoveTimeMetrics(
            total_move_time_us=total_move_time_us,
            events=events,
            timing_model=timing_model,
        )

    def analyze_per_cz_motion(
        self,
        move_mt: ir.Method,
    ) -> tuple[float, float]:
        """Average hops and traveled distance per moving qubit per CZ episode."""
        initial_layout = _infer_initial_qubit_layout(move_mt)
        if initial_layout is None or len(initial_layout) == 0:
            return 0.0, 0.0

        qubit_by_location = {
            location: qubit_id for qubit_id, location in initial_layout.items()
        }

        per_cz_hops: list[float] = []
        per_cz_distance_um: list[float] = []
        episode_stats: dict[int, tuple[int, float]] = {}

        for stmt in move_mt.callable_region.walk():
            if isinstance(stmt, move.Move):
                for lane in stmt.lanes:
                    src, dst = self.arch_spec.get_endpoints(lane)
                    qubit_id = qubit_by_location.pop(src, None)
                    if qubit_id is None:
                        continue
                    qubit_by_location[dst] = qubit_id
                    hop_count, distance_um = episode_stats.get(qubit_id, (0, 0.0))
                    episode_stats[qubit_id] = (
                        hop_count + 1,
                        distance_um + self.move_calc.lane_distance_um(lane),
                    )
                continue

            if isinstance(stmt, move.CZ):
                if len(episode_stats) > 0:
                    for hop_count, distance_um in episode_stats.values():
                        per_cz_hops.append(float(hop_count))
                        per_cz_distance_um.append(distance_um)
                episode_stats = {}

        if len(per_cz_hops) == 0:
            return 0.0, 0.0
        return (
            sum(per_cz_hops) / len(per_cz_hops),
            sum(per_cz_distance_um) / len(per_cz_distance_um),
        )


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


def _compute_event_duration_us(lane_durations_us: list[float]) -> float:
    if len(lane_durations_us) == 0:
        return 0.0
    return max(lane_durations_us)


def _infer_initial_qubit_layout(
    move_mt: ir.Method,
) -> dict[int, Any] | None:
    for stmt in move_mt.callable_region.walk():
        if isinstance(stmt, move.LogicalInitialize):
            return {
                qubit_id: location
                for qubit_id, location in enumerate(stmt.location_addresses)
            }
    return None
