import pytest

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.metrics import Metrics


def _build_metrics() -> Metrics:
    arch_spec = logical.get_arch_spec()
    return Metrics(arch_spec=arch_spec)


def test_metrics_get_lane_duration_us_positive():
    metrics = _build_metrics()
    lanes = tuple(metrics.arch_spec._lane_map.values())
    assert lanes
    for lane in lanes:
        duration = metrics.get_lane_duration_us(lane)
        assert duration > 0.0


def test_metrics_get_lane_duration_cost_bounds_and_anchor():
    metrics = _build_metrics()
    lanes = tuple(metrics.arch_spec._lane_map.values())
    assert lanes
    costs = [metrics.get_lane_duration_cost(lane) for lane in lanes]
    assert all(0.0 <= cost <= 1.0 for cost in costs)
    assert max(costs) == pytest.approx(1.0)


def test_metrics_get_lane_duration_cost_monotonic():
    metrics = _build_metrics()
    lanes = tuple(metrics.arch_spec._lane_map.values())
    assert lanes
    pairs = sorted(
        (metrics.get_lane_duration_us(lane), metrics.get_lane_duration_cost(lane))
        for lane in lanes
    )
    for (_, left_cost), (_, right_cost) in zip(pairs, pairs[1:]):
        assert left_cost <= right_cost + 1e-12


def test_metrics_get_lane_duration_cost_identical_durations_match():
    metrics = _build_metrics()
    lanes = tuple(metrics.arch_spec._lane_map.values())
    assert lanes

    duration_groups: dict[float, list] = {}
    for lane in lanes:
        duration = metrics.get_lane_duration_us(lane)
        duration_groups.setdefault(round(duration, 10), []).append(lane)

    same_duration_group = next(
        (group for group in duration_groups.values() if len(group) >= 2), None
    )
    assert same_duration_group is not None

    baseline = metrics.get_lane_duration_cost(same_duration_group[0])
    for lane in same_duration_group[1:]:
        assert metrics.get_lane_duration_cost(lane) == pytest.approx(baseline)


def test_metrics_caching():
    metrics = _build_metrics()
    lanes = tuple(metrics.arch_spec._lane_map.values())
    lane = lanes[0]
    d1 = metrics.get_lane_duration_us(lane)
    d2 = metrics.get_lane_duration_us(lane)
    assert d1 == d2
