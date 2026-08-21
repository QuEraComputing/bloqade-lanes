import itertools
import math
from dataclasses import dataclass, field
from typing import Any

from bloqade.lanes.bytecode import MotionModel


@dataclass
class MoveMetricCalculator:
    """Move-metric computation: lane durations, distances, and costs.

    The constant-jerk timing model (ramp/jerk/accel constants and the
    move-duration formula) lives in the Rust
    :class:`~bloqade.lanes.bytecode.MotionModel`, the single source of truth
    shared with the move-search solver.  This calculator delegates all timing
    to that object (defaulting to the FLAIR constants) and adds cached lane
    duration / cost lookups on top.  Lives in the ``arch`` package so that
    ``PathFinder`` and heuristics can consume it without pulling in the heavy
    compilation imports of ``Metrics``.

    Pass a non-default ``motion_model`` to price durations with a different
    motion profile.
    """

    arch_spec: Any  # ArchSpec — use Any to avoid circular import
    motion_model: MotionModel = field(default_factory=MotionModel)

    def __post_init__(self) -> None:
        self._lane_duration_cache_us: dict[tuple[Any, float], float] = {}
        self._max_lane_duration_cache_us: dict[float, float] = {}

    def path_segment_distances_um(
        self, path: tuple[tuple[float, float], ...]
    ) -> tuple[float, ...]:
        if len(path) <= 1:
            return ()
        return tuple(
            math.hypot(x1 - x0, y1 - y0)
            for (x0, y0), (x1, y1) in itertools.pairwise(path)
        )

    def get_lane_duration_us(
        self, lane_address: Any, *, amplitude_delta: float = 1.0
    ) -> float:
        """Return lane execution duration in microseconds."""
        normalized_amp = abs(float(amplitude_delta))
        cache_key = (lane_address, normalized_amp)
        if (duration_us := self._lane_duration_cache_us.get(cache_key)) is not None:
            return duration_us

        duration_us = self.motion_model.lane_duration_us(
            self.arch_spec.get_path(lane_address), normalized_amp
        )
        self._lane_duration_cache_us[cache_key] = duration_us
        return duration_us

    def _iter_lane_addresses(self) -> tuple[Any, ...]:
        return tuple(self.arch_spec.iter_all_lanes())

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

    def lane_distance_um(self, lane: Any) -> float:
        """Total distance in µm along a lane's path."""
        path = self.arch_spec.get_path(lane)
        return sum(self.path_segment_distances_um(path))
