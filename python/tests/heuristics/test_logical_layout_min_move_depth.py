"""The exhaustive logical layout estimator and its pipeline opt-in."""

from itertools import pairwise, permutations
from math import hypot

import bloqade.squin as squin
import pytest

import bloqade.gemini as gemini
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.logical.min_move_depth import (
    LogicalLayoutHeuristicMinMoveDepth,
)
from bloqade.lanes.transform.pipeline import LogicalPipeline


def _reference_score(
    homes,
    routes,
    stages: list[tuple[tuple[int, int], ...]],
    placement: tuple[LocationAddress, ...],
) -> tuple[int, float]:
    slots = {home: index for index, home in enumerate(homes)}
    depth = 0
    distance = 0.0
    for stage in stages:
        seen = set()
        for source, target in stage:
            route = routes[slots[placement[source]]][slots[placement[target]]]
            if route.group not in seen:
                seen.add(route.group)
                depth += route.depth
                distance += route.distance
    return depth, distance


@pytest.mark.parametrize(
    "stages,pinned",
    [
        ([((0, 1), (2, 3))], {}),
        ([((0, 1), (2, 3)), ((0, 2), (1, 3))], {}),
        ([((0, 1),), ((1, 2),)], {1: LocationAddress(6, 0)}),
    ],
)
def test_matches_brute_force_with_distance_and_pins(stages, pinned) -> None:
    heuristic = LogicalLayoutHeuristicMinMoveDepth()
    homes = tuple(sorted(heuristic.arch_spec.home_sites))
    qubits = tuple(
        range(4 if any(3 in pair for stage in stages for pair in stage) else 3)
    )
    free = tuple(home for home in homes if home not in pinned.values())
    unpinned = tuple(q for q in qubits if q not in pinned)
    homes, routes = heuristic._geometry()
    candidates = []
    for assignment in permutations(free, len(unpinned)):
        positions = pinned | dict(zip(unpinned, assignment))
        placement = tuple(positions[q] for q in qubits)
        candidates.append(
            (*_reference_score(homes, routes, stages, placement), placement)
        )
    expected = min(candidates)
    actual = heuristic.compute_layout(qubits, stages, pinned)
    assert (_reference_score(homes, routes, stages, actual), actual) == (
        expected[:2],
        expected[2],
    )


def test_left_and_right_buses_are_separate_but_cross_pairs_group() -> None:
    heuristic = LogicalLayoutHeuristicMinMoveDepth()
    homes, routes = heuristic._geometry()
    by_word = {home.word_id: index for index, home in enumerate(homes)}

    left = routes[by_word[0]][by_word[4]]
    right = routes[by_word[2]][by_word[6]]
    assert left.depth == right.depth == 1
    assert left.group != right.group

    cross_a = routes[by_word[0]][by_word[6]]
    cross_b = routes[by_word[4]][by_word[10]]
    assert cross_a.depth == cross_b.depth == 2
    assert cross_a.group == cross_b.group


def test_route_distance_uses_exact_architecture_polylines() -> None:
    heuristic = LogicalLayoutHeuristicMinMoveDepth()
    homes, routes = heuristic._geometry()
    by_word = {home.word_id: index for index, home in enumerate(homes)}
    arch = heuristic.arch_spec

    def lane_length(source_word: int, target_word: int) -> float:
        source = LocationAddress(source_word, 0)
        target = LocationAddress(target_word, 0)
        lane = arch.get_lane_address(source, target)
        assert lane is not None
        return sum(
            hypot(x1 - x0, y1 - y0)
            for (x0, y0), (x1, y1) in pairwise(arch.get_path(lane))
        )

    assert routes[by_word[0]][by_word[4]].distance == lane_length(0, 5)
    assert routes[by_word[0]][by_word[6]].distance == lane_length(0, 5) + lane_length(
        5, 7
    )


def test_inactive_qubits_fill_lexicographically_and_pins_hold() -> None:
    heuristic = LogicalLayoutHeuristicMinMoveDepth()
    homes = tuple(sorted(heuristic.arch_spec.home_sites))
    pinned = {2: homes[4]}
    assert heuristic.compute_layout((0, 1, 2, 3), [], pinned) == (
        homes[0],
        homes[1],
        homes[4],
        homes[2],
    )


def test_pipeline_exposes_opt_in_without_changing_default() -> None:
    arch = get_arch_spec()
    heuristic = LogicalLayoutHeuristicMinMoveDepth(arch_spec=arch)
    assert (
        LogicalPipeline(
            arch_spec=arch, layout_heuristic=heuristic
        ).resolved_layout_heuristic
        is heuristic
    )
    assert type(LogicalPipeline().resolved_layout_heuristic) is not type(heuristic)


def test_opt_in_pipeline_compiles_a_logical_cz() -> None:
    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.cz(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    out = LogicalPipeline(layout_heuristic=LogicalLayoutHeuristicMinMoveDepth()).emit(
        kernel
    )
    assert any(isinstance(stmt, move.Fill) for stmt in out.callable_region.walk())
