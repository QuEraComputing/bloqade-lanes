import math
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class MoveMetricCalculator:
    """Move-metric computation: lane durations, distances, and costs.

    Owns timing constants extracted from bloqade-flair and provides
    cached lane duration / cost lookups.  Lives in the ``layout``
    package so that ``PathFinder`` and heuristics can consume it
    without pulling in the heavy compilation imports of ``Metrics``.
    """

    arch_spec: Any  # ArchSpec — use Any to avoid circular import

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

    def lane_distance_um(self, lane: Any) -> float:
        """Total distance in µm along a lane's path."""
        path = self.arch_spec.get_path(lane)
        return sum(self.path_segment_distances_um(path))
