"""Timing, legal batching, search bounds and pipeline integration."""

import json
from itertools import pairwise, permutations
from math import fsum, inf

import bloqade.squin as squin
import pytest
from bloqade.rewrite.passes import AggressiveUnroll
from kirin import interp, rewrite

from bloqade.lanes.analysis.atom.atom_state_data import AtomStateData
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import MotionModel
from bloqade.lanes.bytecode._native import ArchSpec as RustArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move, steane713
from bloqade.lanes.heuristics.logical import (
    LogicalLayoutHeuristic,
    LogicalLayoutHeuristicMinMoveTime,
)
from bloqade.lanes.rewrite.qubit_count import InsertQubitCount
from bloqade.lanes.transform.pipeline import LogicalPipeline


def _small_arch(buses, paths=None, aod_capacity=None):
    """Four home/staging pairs, with explicit bus membership and path overrides."""
    spec = json.loads(get_arch_spec().to_json())
    spec["words"] = spec["words"][:8]
    zone = spec["zones"][0]
    zone["grid"]["y_spacing"] = [10.0]
    zone["word_buses"] = [{"src": list(src), "dst": list(dst)} for src, dst in buses]
    zone["entangling_pairs"] = zone["entangling_pairs"][:4]
    spec["paths"] = []
    if aod_capacity is not None:
        spec["aod_capacity"] = list(aod_capacity)
    arch = ArchSpec(RustArchSpec.from_json_validated(json.dumps(spec)))
    for (source, target), points in (paths or {}).items():
        lane = arch.get_lane_address(
            LocationAddress(source, 0), LocationAddress(target, 0)
        )
        assert lane is not None
        spec["paths"].append(
            {"lane": f"0x{lane._inner.encode():016X}", "waypoints": points}
        )
    return ArchSpec(RustArchSpec.from_json_validated(json.dumps(spec)))


def _scorer(arch):
    return LogicalLayoutHeuristicMinMoveTime(arch_spec=arch)._prepare()


def _assert_executable(scorer, occupied, schedule):
    initial = {i: loc for i, loc in enumerate(scorer.homes) if occupied & (1 << i)}
    state = AtomStateData.new(initial)
    for layer in schedule.layers:
        state = state.apply_validated(state.validate_moves(layer, scorer.arch))
    assert dict(state.qubit_to_locations) == initial
    # Independent pricing of the actual sequential program, including return.
    assert schedule.duration_us == pytest.approx(
        fsum(
            max(scorer.metrics.get_lane_duration_us(lane) for lane in layer)
            for layer in schedule.layers
        )
    )


def test_paths_minimize_time_not_distance_or_hop_count():
    # Direct 0->3 is shorter in distance, but has many stop/start segments.
    arch = _small_arch(
        [([0], [3]), ([0], [1]), ([1], [5]), ([5], [3])],
        {(0, 3): [(i / 2, 0.0) for i in range(25)]},
    )
    scorer = _scorer(arch)
    route = scorer.routes[0][1]
    assert route is not None
    assert [(s.source.word_id, s.target.word_id) for s in route.steps] == [
        (0, 1),
        (1, 5),
        (5, 3),
    ]
    direct = arch.get_lane_address(LocationAddress(0, 0), LocationAddress(3, 0))
    assert direct is not None
    assert sum(
        scorer.metrics.lane_distance_um(s.lane) for s in route.steps
    ) > scorer.metrics.lane_distance_um(direct)
    assert sum(
        s.duration_us for s in route.steps
    ) < scorer.metrics.get_lane_duration_us(direct)
    schedule = scorer.schedule(0b0011, ((0, 1),))
    assert schedule is not None and len(schedule.layers) == 6
    _assert_executable(scorer, 0b0011, schedule)


def test_parallel_batch_uses_longest_lane_and_prices_return_separately():
    arch = _small_arch(
        [([0, 2], [5, 7])],
        {
            (0, 5): [(0, 0), (-10, 0), (-10, 10), (2, 10)],
            (7, 2): [(12, 10), (30, 10), (30, 0), (10, 0)],
        },
    )
    scorer = _scorer(arch)
    schedule = scorer.schedule(0b1111, ((0, 2), (1, 3)))
    assert schedule is not None
    assert [len(layer) for layer in schedule.layers] == [2, 2]
    _assert_executable(scorer, 0b1111, schedule)
    assert scorer.schedule(0b1111, ((1, 3), (0, 2))) == schedule
    a, b = scorer.routes[0][2], scorer.routes[1][3]
    assert a is not None and b is not None
    assert schedule.duration_us == pytest.approx(
        max(a.steps[0].duration_us, b.steps[0].duration_us)
        + max(a.steps[0].return_duration_us, b.steps[0].return_duration_us)
    )


@pytest.mark.parametrize(
    "buses",
    [
        [([0], [5]), ([2], [7])],  # different buses
        [([0, 7], [5, 2])],  # same bus, opposite directions
    ],
)
def test_distinct_buses_or_directions_cannot_batch(buses):
    scorer = _scorer(_small_arch(buses))
    schedule = scorer.schedule(0b1111, ((0, 2), (1, 3)))
    assert schedule is not None
    assert [len(layer) for layer in schedule.layers] == [1, 1, 1, 1]
    _assert_executable(scorer, 0b1111, schedule)


def test_aod_capacity_splits_same_bus_and_direction():
    scorer = _scorer(_small_arch([([0, 2], [5, 7])], aod_capacity=(1, 1)))
    schedule = scorer.schedule(0b1111, ((0, 2), (1, 3)))
    assert schedule is not None
    assert [len(layer) for layer in schedule.layers] == [1, 1, 1, 1]
    _assert_executable(scorer, 0b1111, schedule)


def test_batch_can_vacate_another_movers_destination():
    arch = _small_arch([([0, 4], [4, 7]), ([4], [3])])
    spec = json.loads(arch.to_json())
    spec["zones"][0]["grid"]["y_spacing"] = [10.0, 10.0]
    spec["words"][7]["sites"] = [[0, 2]]
    scorer = _scorer(ArchSpec(RustArchSpec.from_json_validated(json.dumps(spec))))
    # 0->4 is blocked on its own; jointly with 4->7 both are executable.
    schedule = scorer.schedule(0b1111, ((0, 1), (2, 3)))
    assert schedule is not None
    assert [len(layer) for layer in schedule.layers] == [2, 1, 1, 2]
    _assert_executable(scorer, 0b1111, schedule)


def test_only_ready_hops_can_batch():
    # The two cross-column routes can share their second hop only after both
    # different first-hop buses have executed. The greedy policy may execute
    # one crossing early, but must never move an atom before its path is ready.
    scorer = _scorer(get_arch_spec())
    pairs = ((0, 3), (4, 7))
    occupied = sum(1 << i for pair in pairs for i in pair)
    schedule = scorer.schedule(occupied, pairs)
    assert schedule is not None
    _assert_executable(scorer, occupied, schedule)
    forward = schedule.layers[: len(schedule.layers) // 2]
    for source, target in pairs:
        route = scorer.routes[source][target]
        assert route is not None
        layer_indices = [
            next(i for i, layer in enumerate(forward) if s.lane in layer)
            for s in route.steps
        ]
        assert all(a < b for a, b in pairwise(layer_indices))


def test_blocked_fixed_paths_are_rejected_and_pinned_failure_is_clear():
    # Each atom's first step requires the other atom to leave on a different bus.
    arch = _small_arch([([0], [4]), ([4], [3]), ([0], [7])])
    scorer = _scorer(arch)
    assert scorer.schedule(0b1111, ((0, 1), (2, 3))) is None
    with pytest.raises(
        ValueError, match="no home layout admits a legal greedy schedule"
    ):
        LogicalLayoutHeuristicMinMoveTime(arch_spec=arch).compute_layout(
            (0, 1, 2, 3), [((0, 1), (2, 3))], dict(enumerate(scorer.homes))
        )


def test_spectator_placement_is_searched_because_it_can_block_a_route():
    arch = _small_arch([([0], [4]), ([4], [3])])
    heuristic = LogicalLayoutHeuristicMinMoveTime(arch_spec=arch)
    homes = tuple(sorted(arch.home_sites))
    assert heuristic.compute_layout(
        (0, 1, 2), [((0, 1),)], {0: homes[0], 1: homes[1]}
    ) == (homes[0], homes[1], homes[3])


def test_sparse_ids_and_multiple_spectators_keep_lexicographic_order():
    h = LogicalLayoutHeuristicMinMoveTime()
    homes = tuple(sorted(h.arch_spec.home_sites))
    layout = h.compute_layout((9, 2, 17, 1), [((2, 9),)], {2: homes[0], 9: homes[2]})
    assert layout == (homes[1], homes[0], homes[2], homes[3])


def test_unreachable_routes_are_not_given_zero_cost():
    arch = _small_arch([])
    scorer = _scorer(arch)
    assert scorer.routes[0][1] is None
    with pytest.raises(ValueError, match="no home layout"):
        LogicalLayoutHeuristicMinMoveTime(arch_spec=arch).compute_layout(
            (0, 1), [((0, 1),)]
        )


def test_motion_model_is_used_for_both_path_selection_and_scoring():
    arch = _small_arch([([0], [3])])
    normal = _scorer(arch)
    slower = LogicalLayoutHeuristicMinMoveTime(
        arch_spec=arch, motion_model=MotionModel(max_ramp_us=0.1)
    )._prepare()
    a = normal.schedule(0b0011, ((0, 1),))
    b = slower.schedule(0b0011, ((0, 1),))
    assert a is not None and b is not None
    assert b.duration_us - a.duration_us == pytest.approx(20.0)


@pytest.mark.parametrize(
    "stages,pinned",
    [
        ([((0, 1), (2, 3))], {}),
        ([((0, 1), (2, 3)), ((0, 2), (1, 3))], {}),
        ([((0, 1),), ((1, 2),)], {1: LocationAddress(6, 0)}),
    ],
)
def test_branch_bound_matches_unpruned_enumeration(stages, pinned):
    h = LogicalLayoutHeuristicMinMoveTime()
    scorer = h._prepare()
    qubits = tuple(sorted({q for stage in stages for pair in stage for q in pair}))
    index = {home: i for i, home in enumerate(scorer.homes)}
    free = tuple(home for home in scorer.homes if home not in pinned.values())
    unpinned = tuple(q for q in qubits if q not in pinned)
    best = None
    for assignment in permutations(free, len(unpinned)):
        layout = pinned | dict(zip(unpinned, assignment))
        slots = tuple(index[layout[q]] for q in qubits)
        occupied = sum(1 << slot for slot in slots)
        costs = []
        for stage in stages:
            schedule = scorer.schedule(
                occupied, tuple((slots[s], slots[t]) for s, t in stage)
            )
            costs.append(inf if schedule is None else schedule.duration_us)
        score = fsum(costs)
        # Same tolerance convention for physically equal schedules, independent
        # enumeration without any partial-layout pruning.
        if (
            best is None
            or score < best[0] - 1e-9
            or (abs(score - best[0]) <= 1e-9 and slots < best[1])
        ):
            best = score, slots
    assert best is not None
    assert h.compute_layout(qubits, stages, pinned) == tuple(
        scorer.homes[i] for i in best[1]
    )


def test_no_interactions_fill_in_address_order_and_pins_hold():
    h = LogicalLayoutHeuristicMinMoveTime()
    homes = tuple(sorted(h.arch_spec.home_sites))
    assert h.compute_layout((3, 0, 2, 1), [], {2: homes[4]}) == (
        homes[0],
        homes[1],
        homes[4],
        homes[2],
    )


@pytest.mark.parametrize(
    "stages,message",
    [
        ([((0, 2),)], "absent from all_qubits"),
        ([((0, 0),)], "cannot pair a qubit with itself"),
        ([((0, 1), (1, 0))], "disjoint qubit pairs"),
    ],
)
def test_invalid_stages(stages, message):
    with pytest.raises(ValueError, match=message):
        LogicalLayoutHeuristicMinMoveTime().compute_layout((0, 1), stages)


def test_input_validation():
    h = LogicalLayoutHeuristicMinMoveTime()
    with pytest.raises(interp.InterpreterError, match="exceeds maximum supported"):
        h.compute_layout(tuple(range(11)), [])
    with pytest.raises(ValueError, match="unique qubit IDs"):
        h.compute_layout((0, 0), [])
    with pytest.raises(ValueError, match="pinned addresses must be unique"):
        h.compute_layout(
            (0, 1), [], {0: LocationAddress(0, 0), 1: LocationAddress(0, 0)}
        )
    with pytest.raises(ValueError, match="not in all_qubits"):
        h.compute_layout((0,), [], {1: LocationAddress(0, 0)})
    with pytest.raises(ValueError, match="not valid home"):
        h.compute_layout((0,), [], {0: LocationAddress(1, 0)})


def test_missing_staging_partner_is_rejected(monkeypatch):
    h = LogicalLayoutHeuristicMinMoveTime()
    monkeypatch.setattr(h.arch_spec, "get_cz_partner", lambda _home: None)
    with pytest.raises(ValueError, match="CZ staging partner"):
        h._prepare()


def test_opt_in_heuristic_and_pipeline_default():
    h = LogicalLayoutHeuristicMinMoveTime()
    assert (
        LogicalPipeline(
            arch_spec=h.arch_spec, layout_heuristic=h
        ).resolved_layout_heuristic
        is h
    )
    assert type(LogicalPipeline().resolved_layout_heuristic) is LogicalLayoutHeuristic


def test_opt_in_pipeline_compiles_a_logical_cz():
    @squin.kernel.add(steane713)
    def kernel():
        reg = squin.qalloc(2)
        squin.cz(reg[0], reg[1])
        steane713.terminal_measure(reg)

    AggressiveUnroll(kernel.dialects).fixpoint(kernel)
    rewrite.Walk(InsertQubitCount(7)).rewrite(kernel.code)

    out = LogicalPipeline(layout_heuristic=LogicalLayoutHeuristicMinMoveTime()).emit(
        kernel
    )
    assert any(isinstance(stmt, move.Fill) for stmt in out.callable_region.walk())
