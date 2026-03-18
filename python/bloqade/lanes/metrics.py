import math
from dataclasses import dataclass
from typing import Any, ClassVar

from bloqade.analysis.fidelity import FidelityAnalysis, FidelityRange
from kirin import ir

from bloqade.lanes.analysis.placement.strategy import PlacementStrategyABC
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.logical_mvp import transversal_rewrites
from bloqade.lanes.noise_model import generate_simple_noise_model
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

    Owns all timing constants, lane duration computation, and kernel
    analysis methods. Construct once at a pipeline entry point and
    thread through to all consumers (PathFinder, heuristics, etc.).
    """

    arch_spec: Any  # ArchSpec — use Any to avoid circular import
    noise_model: NoiseModelABC | None = None

    _FLAIR_MAX_RAMP_US: ClassVar[float] = 0.2
    _FLAIR_MAX_JERK_UM_PER_US3: ClassVar[float] = 0.0004
    _FLAIR_MAX_ACCEL_UM_PER_US2: ClassVar[float] = 0.0015

    def __post_init__(self) -> None:
        self._lane_duration_cache_us: dict[tuple[Any, float], float] = {}
        self._max_lane_duration_cache_us: dict[float, float] = {}

    def path_segment_distances_um(
        self, path: tuple[tuple[float, float], ...]
    ) -> tuple[float, ...]:
        if len(path) <= 1:
            return ()
        return tuple(
            math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:])
        )

    def _const_jerk_min_duration_us(self, max_dist_um: float) -> float:
        max_dist_um = abs(max_dist_um)
        if max_dist_um < 1e-8:
            return 0.0

        t1 = self._FLAIR_MAX_ACCEL_UM_PER_US2 / self._FLAIR_MAX_JERK_UM_PER_US3
        a = self._FLAIR_MAX_JERK_UM_PER_US3 * t1
        b = 3 * self._FLAIR_MAX_JERK_UM_PER_US3 * t1**2
        c = 2 * self._FLAIR_MAX_JERK_UM_PER_US3 * t1**3 - max_dist_um
        if c >= 0:
            t1_jerk = (max_dist_um / (2 * self._FLAIR_MAX_JERK_UM_PER_US3)) ** (1 / 3)
            return 4 * t1_jerk

        discriminant = b**2 - 4 * a * c
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)
        return 4 * t1 + 2 * t2

    def get_lane_duration_us(
        self, lane_address: Any, *, amplitude_delta: float = 1.0
    ) -> float:
        """Return lane execution duration in microseconds."""
        normalized_amp = abs(float(amplitude_delta))
        cache_key = (lane_address, normalized_amp)
        if (duration_us := self._lane_duration_cache_us.get(cache_key)) is not None:
            return duration_us

        segment_distances = self.path_segment_distances_um(
            self.arch_spec.get_path(lane_address)
        )
        ramp_time_us = normalized_amp / self._FLAIR_MAX_RAMP_US
        duration_us = (
            ramp_time_us
            + sum(self._const_jerk_min_duration_us(dist) for dist in segment_distances)
            + ramp_time_us
        )
        self._lane_duration_cache_us[cache_key] = duration_us
        return duration_us

    def _iter_lane_addresses(self) -> tuple[Any, ...]:
        return tuple(self.arch_spec._lane_map.values())

    def _max_lane_duration_us(self, *, amplitude_delta: float = 1.0) -> float:
        normalized_amp = abs(float(amplitude_delta))
        if (
            max_duration_us := self._max_lane_duration_cache_us.get(normalized_amp)
        ) is not None:
            return max_duration_us

        lane_addresses = self._iter_lane_addresses()
        if len(lane_addresses) == 0:
            max_duration_us = 0.0
        else:
            max_duration_us = max(
                self.get_lane_duration_us(lane, amplitude_delta=normalized_amp)
                for lane in lane_addresses
            )
        self._max_lane_duration_cache_us[normalized_amp] = max_duration_us
        return max_duration_us

    def get_lane_duration_cost(
        self, lane_address: Any, *, amplitude_delta: float = 1.0
    ) -> float:
        """Return normalized lane duration cost in [0, 1]."""
        max_duration_us = self._max_lane_duration_us(amplitude_delta=amplitude_delta)
        if max_duration_us <= 0.0:
            return 0.0
        lane_duration_us = self.get_lane_duration_us(
            lane_address, amplitude_delta=amplitude_delta
        )
        return min(1.0, max(0.0, lane_duration_us / max_duration_us))

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
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(),
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

    def _lane_distance_um(self, lane) -> float:
        path = self.arch_spec.get_path(lane)
        return sum(self.path_segment_distances_um(path))

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
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(),
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
            layout_heuristic=logical_layout.LogicalLayoutHeuristic(),
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
                segment_distances_um = list(self.path_segment_distances_um(path))
                segment_durations_us = [
                    self._const_jerk_min_duration_us(d) for d in segment_distances_um
                ]
                normalized_amp = abs(float(flair_amplitude_delta))
                ramp_time_us = normalized_amp / self._FLAIR_MAX_RAMP_US
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
                        distance_um + self._lane_distance_um(lane),
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
