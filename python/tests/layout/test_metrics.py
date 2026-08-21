import itertools

import pytest

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.metrics import MoveMetricCalculator
from bloqade.lanes.bytecode import MotionModel


def _build_move_calc() -> MoveMetricCalculator:
    arch_spec = logical.get_arch_spec()
    return MoveMetricCalculator(arch_spec=arch_spec)


def test_motion_model_pins_flair_defaults_and_known_durations():
    """Guard the shared FLAIR model reachable from Python.

    Pins the default constants and known durations so an accidental edit to
    the Rust constants/formula — or to the ``bytecode.MotionModel`` binding
    wiring — is caught by the Python suite, not only the Rust tests.
    """
    model = MotionModel()
    assert (
        model.max_ramp_us,
        model.max_jerk_um_per_us3,
        model.max_accel_um_per_us2,
    ) == (0.2, 0.0004, 0.0015)
    assert MotionModel.flair() == model

    # Known constant-jerk duration for a 5 µm move (quadratic-branch value).
    assert model.const_jerk_min_duration_us(5.0) == pytest.approx(
        119.280930201974, rel=1e-12
    )
    # Lane duration = pick ramp + one 5 µm segment + drop ramp.
    ramp_us = 1.0 / 0.2
    assert model.lane_duration_us([(0.0, 0.0), (3.0, 4.0)], 1.0) == pytest.approx(
        ramp_us + model.const_jerk_min_duration_us(5.0) + ramp_us, rel=1e-12
    )


def test_metrics_get_lane_duration_us_positive():
    move_calc = _build_move_calc()
    lanes = tuple(move_calc.arch_spec.iter_all_lanes())
    assert lanes
    for lane in lanes:
        duration = move_calc.get_lane_duration_us(lane)
        assert duration > 0.0


def test_metrics_get_lane_duration_cost_bounds_and_anchor():
    move_calc = _build_move_calc()
    lanes = tuple(move_calc.arch_spec.iter_all_lanes())
    assert lanes
    costs = [move_calc.get_lane_duration_cost(lane) for lane in lanes]
    assert all(0.0 <= cost <= 1.0 for cost in costs)
    assert max(costs) == pytest.approx(1.0)


def test_metrics_get_lane_duration_cost_monotonic():
    move_calc = _build_move_calc()
    lanes = tuple(move_calc.arch_spec.iter_all_lanes())
    assert lanes
    pairs = sorted(
        (move_calc.get_lane_duration_us(lane), move_calc.get_lane_duration_cost(lane))
        for lane in lanes
    )
    for (_, left_cost), (_, right_cost) in itertools.pairwise(pairs):
        assert left_cost <= right_cost + 1e-12


def test_metrics_get_lane_duration_cost_identical_durations_match():
    move_calc = _build_move_calc()
    lanes = tuple(move_calc.arch_spec.iter_all_lanes())
    assert lanes

    duration_groups: dict[float, list] = {}
    for lane in lanes:
        duration = move_calc.get_lane_duration_us(lane)
        duration_groups.setdefault(round(duration, 10), []).append(lane)

    same_duration_group = next(
        (group for group in duration_groups.values() if len(group) >= 2), None
    )
    assert same_duration_group is not None

    baseline = move_calc.get_lane_duration_cost(same_duration_group[0])
    for lane in same_duration_group[1:]:
        assert move_calc.get_lane_duration_cost(lane) == pytest.approx(baseline)


def test_metrics_caching():
    move_calc = _build_move_calc()
    lanes = tuple(move_calc.arch_spec.iter_all_lanes())
    lane = lanes[0]
    d1 = move_calc.get_lane_duration_us(lane)
    d2 = move_calc.get_lane_duration_us(lane)
    assert d1 == d2
