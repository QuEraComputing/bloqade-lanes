"""The typed placement and bound-statistics surface of the search bindings.

Covers ``*CzPlacement.place`` and its ``PlacementResult`` / ``CandidateAttempt``,
``SolveResult.bound_stats`` as ``BoundStats`` or ``None``, and the typed
``SearchConfigError`` subclasses.
"""

from __future__ import annotations

import pytest

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode._native import (
    BoundStats,
    CandidateAttempt,
    EntanglingOptions,
    EntropyOptions,
    LocationAddress,
    LooseGoalCzPlacement,
    MoveSearch,
    PlacementResult,
    SearchEngine,
    SingleHeuristicCzPlacement,
    SolveStatus,
    TargetSolver,
)
from bloqade.lanes.bytecode.exceptions import (
    DuplicateOccupancyError,
    DuplicateQubitIdError,
    DuplicateTargetLocationError,
    DuplicateTargetQubitError,
    SearchConfigError,
)


def _engine() -> SearchEngine:
    return SearchEngine.from_arch_spec(logical.get_arch_spec()._inner)


def _loc(word: int) -> LocationAddress:
    # Native order is (zone, word, site); the logical spec has one site per word.
    return LocationAddress(0, word, 0)


def test_single_heuristic_place_reports_typed_attempts():
    engine = _engine()
    placement = SingleHeuristicCzPlacement(TargetSolver(engine, MoveSearch.astar(1.0)))
    placed = placement.place({0: _loc(0), 1: _loc(3)}, [(0, 1)], [], 2000)

    assert isinstance(placed, PlacementResult)
    assert placed.result.status == SolveStatus.SOLVED
    assert placed.candidates_tried == len(placed.attempts) >= 1
    assert all(isinstance(a, CandidateAttempt) for a in placed.attempts)
    assert placed.attempts[-1].status == SolveStatus.SOLVED
    assert placed.chosen == placed.attempts[-1].candidate_index
    assert placed.total_expansions == sum(a.nodes_expanded for a in placed.attempts)
    assert all(a.score is None for a in placed.attempts)


def test_loose_goal_place_routes_once():
    engine = _engine()
    placement = LooseGoalCzPlacement(engine, MoveSearch.astar(1.0), EntanglingOptions())
    placed = placement.place({0: _loc(0), 1: _loc(3)}, [(0, 1)], [], 2000)

    assert placed.chosen is None
    assert placed.attempts == []
    assert placed.total_expansions == placed.result.nodes_expanded


def test_bound_stats_is_typed_when_bounded_and_none_otherwise():
    engine = _engine()
    initial = {0: _loc(0), 1: _loc(1)}
    target = {0: _loc(3), 1: _loc(4)}

    unbounded = TargetSolver(engine, MoveSearch.entropy()).solve(
        initial, target, [], 2000
    )
    assert unbounded.bound_stats is None

    bounded_search = MoveSearch.entropy().with_entropy_options(
        EntropyOptions(completion_bound="weighted_distance")
    )
    bounded = TargetSolver(engine, bounded_search).solve(initial, target, [], 2000)
    stats = bounded.bound_stats
    assert isinstance(stats, BoundStats)
    assert stats.root_lower_bound > 0.0
    assert isinstance(stats.cuts_by_h, int)
    if bounded.status == SolveStatus.SOLVED:
        assert stats.incumbent_cost == bounded.cost
        assert stats.optimality_gap is not None


def test_an_invalid_request_raises_its_typed_error():
    """Each ``ConfigError`` variant arrives as its own ``SearchConfigError``
    subclass, still a ``ValueError``, carrying the variant's fields."""
    solver = TargetSolver(_engine(), MoveSearch.astar(1.0))

    with pytest.raises(DuplicateTargetLocationError) as info:
        solver.solve({0: _loc(0), 1: _loc(1)}, {0: _loc(3), 1: _loc(3)}, [], 100)
    assert isinstance(info.value, SearchConfigError)
    assert isinstance(info.value, ValueError)
    assert info.value.qubits in ((0, 1), (1, 0))

    with pytest.raises(DuplicateOccupancyError) as occupancy:
        TargetSolver(_engine(), MoveSearch.astar(1.0)).solve(
            {0: _loc(0), 1: _loc(0)}, {0: _loc(3), 1: _loc(4)}, [], 100
        )
    assert occupancy.value.location == _loc(0).encode()


@pytest.mark.parametrize(
    ("cls", "args", "fields"),
    [
        (DuplicateQubitIdError, ("m", 3), {"qubit_id": 3}),
        (DuplicateTargetQubitError, ("m", 4), {"qubit_id": 4}),
        (
            DuplicateTargetLocationError,
            ("m", 9, (0, 1)),
            {"location": 9, "qubits": (0, 1)},
        ),
        (DuplicateOccupancyError, ("m", 9, (0, 1)), {"location": 9, "qubits": (0, 1)}),
    ],
)
def test_search_config_errors_carry_their_fields(cls, args, fields):
    """Every subclass is a ``SearchConfigError`` and a ``ValueError`` carrying
    its variant's fields. Two of them (duplicate qubit ids) cannot be provoked
    through the binding, whose dict arguments cannot repeat a key, so the
    classes are checked directly."""
    err = cls(*args)
    assert isinstance(err, SearchConfigError) and isinstance(err, ValueError)
    assert str(err) == "m"
    for name, value in fields.items():
        assert getattr(err, name) == value


def test_single_heuristic_routes_caller_supplied_candidates_in_order():
    """``candidates`` replaces the default generator: invalid ones are skipped
    before routing, and the rest are tried in the order given."""
    engine = _engine()
    placement = SingleHeuristicCzPlacement(TargetSolver(engine, MoveSearch.astar(1.0)))
    initial = {0: _loc(0), 1: _loc(3)}
    default = placement.place(initial, [(0, 1)], [], 2000)
    assert default.result.status == SolveStatus.SOLVED
    good = default.result.goal_config

    # The starting placement does not put the pair on partner sites, so it
    # fails validation and is never routed.
    placed = placement.place(initial, [(0, 1)], [], 2000, candidates=[initial, good])
    assert placed.result.status == SolveStatus.SOLVED
    assert placed.chosen == 1
    assert [a.candidate_index for a in placed.attempts] == [1]

    nothing = placement.place(initial, [(0, 1)], [], 2000, candidates=[])
    assert nothing.result.status == SolveStatus.UNSOLVABLE
    assert nothing.attempts == []
