"""Python surface of the branch-and-bound strategy.

Round-trips of the new option bundles, the ``MoveSearch.branch_and_bound``
factory, and end-to-end solves whose verdicts are proofs: an optimal plan on
the two-zone fixture, a proven infeasibility, and a spec the exhaustive
generator's preconditions refuse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import (
    ArchSpec as RustArchSpec,
    BnbOptions,
    EntropyOptions,
    MoveSearch,
    SearchEngine,
    SearchStrategy,
    SolveOptions,
)
from bloqade.lanes.bytecode.encoding import LocationAddress

from .test_zone_bus_search import _TWO_ZONE_ARCH_JSON

_FULL_JSON = Path(__file__).resolve().parents[3] / "examples" / "arch" / "full.json"


def _two_zone_engine() -> SearchEngine:
    return SearchEngine.from_arch_spec(RustArchSpec.from_json(_TWO_ZONE_ARCH_JSON))


def _proving_search() -> MoveSearch:
    return MoveSearch.branch_and_bound(
        bnb_options=BnbOptions(widen_after_incumbent=255)
    )


def test_bnb_options_round_trip_and_defaults():
    default = BnbOptions()
    assert (default.frontier, default.ordering, default.schedule) == (
        "lifo",
        "hop_sum",
        "entropy_then_exhaustive",
    )
    assert (default.widen_order, default.widen_after_incumbent) == (
        "stage_then_depth",
        0,
    )
    custom = BnbOptions(
        frontier="ids",
        ordering="bound",
        schedule="heuristic_then_exhaustive",
        widen_order="best_bound",
        widen_after_incumbent=255,
    )
    assert custom.frontier == "ids"
    assert custom.ordering == "bound"
    assert custom.schedule == "heuristic_then_exhaustive"
    assert custom.widen_order == "best_bound"
    assert custom.widen_after_incumbent == 255
    assert "ids" in repr(custom)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frontier": "bfs"},
        {"ordering": "sum"},
        {"schedule": "exhaustive"},
        {"widen_order": "depth"},
        {"widen_after_incumbent": 256},
    ],
)
def test_bnb_options_reject_unknown_values(kwargs):
    with pytest.raises(ValueError):
        BnbOptions(**kwargs)


def test_solve_and_entropy_options_carry_the_new_knobs():
    opts = SolveOptions(
        strategy=SearchStrategy.CASCADE_IDS,
        aod_capacity=(2, 3),
        cascade_refine="branch_and_bound",
    )
    assert opts.aod_capacity == (2, 3)
    assert opts.cascade_refine == "branch_and_bound"
    assert opts.strategy == SearchStrategy.CASCADE_IDS
    assert SolveOptions().aod_capacity is None
    assert SolveOptions().cascade_refine == "astar"
    with pytest.raises(ValueError):
        SolveOptions(aod_capacity=(0, 2))
    with pytest.raises(ValueError):
        SolveOptions(cascade_refine="dijkstra")

    eopts = EntropyOptions(objective="weighted_duration", tau=2.5)
    assert (eopts.objective, eopts.tau) == ("weighted_duration", 2.5)
    assert EntropyOptions().objective == "uniform"
    assert EntropyOptions().tau is None
    with pytest.raises(ValueError):
        EntropyOptions(objective="fidelity")
    with pytest.raises(ValueError):
        EntropyOptions(tau=0.0)


def test_move_search_branch_and_bound_is_bounded_out_of_the_box():
    ms = MoveSearch.branch_and_bound()
    assert ms.strategy == SearchStrategy.BRANCH_AND_BOUND
    assert ms.bnb_options.schedule == "entropy_then_exhaustive"
    # The strategy is forced even when the options name another one.
    forced = MoveSearch.branch_and_bound(SolveOptions(strategy=SearchStrategy.ASTAR))
    assert forced.strategy == SearchStrategy.BRANCH_AND_BOUND
    replaced = ms.with_bnb_options(BnbOptions(frontier="dfs"))
    assert replaced.bnb_options.frontier == "dfs"
    assert ms.bnb_options.frontier == "lifo", "with_bnb_options returns a copy"


def test_branch_and_bound_solves_and_proves_across_the_zone_bus():
    engine = _two_zone_engine()
    mem = LocationAddress(1, 0, 1)
    gate = LocationAddress(0, 0, 0)
    solver = _native.TargetSolver(engine, _proving_search())
    result = solver.solve({0: mem._inner}, {0: gate._inner}, [], None)
    assert result.status == "solved"
    assert result.proven is True
    assert result.termination == "exhausted_proof"
    assert result.cost == 1.0
    assert sum(result.stage_expansions) == result.nodes_expanded
    assert result.plan_stage is not None
    assert "proven=true" in repr(result)


def test_branch_and_bound_proves_infeasibility():
    # Two atoms on the two-site fixture asked to swap: no plan exists, and a
    # complete schedule drained with unlimited widening is the proof.
    engine = _two_zone_engine()
    mem = LocationAddress(1, 0, 1)
    gate = LocationAddress(0, 0, 0)
    solver = _native.TargetSolver(engine, _proving_search())
    result = solver.solve(
        {0: mem._inner, 1: gate._inner}, {0: gate._inner, 1: mem._inner}, [], None
    )
    assert result.status == "unsolvable"
    assert result.proven is True
    assert result.termination == "exhausted_proof"

    # An entropy-only schedule can never prove anything.
    unproven = _native.TargetSolver(
        engine,
        MoveSearch.branch_and_bound(
            bnb_options=BnbOptions(schedule="entropy_only", widen_after_incumbent=255)
        ),
    ).solve({0: mem._inner, 1: gate._inner}, {0: gate._inner, 1: mem._inner}, [], None)
    assert unproven.status == "unsolvable"
    assert unproven.proven is False
    assert unproven.termination == "exhausted"


def test_branch_and_bound_refuses_a_spec_that_fails_the_preconditions():
    engine = SearchEngine.from_json(_FULL_JSON.read_text())
    solver = _native.TargetSolver(engine, _proving_search())
    a = LocationAddress(0, 0, 0)
    b = LocationAddress(0, 1, 0)
    with pytest.raises(ValueError, match="P1"):
        solver.solve({0: a._inner}, {0: b._inner}, [], 10)


def test_other_strategies_report_no_proof():
    engine = _two_zone_engine()
    mem = LocationAddress(1, 0, 1)
    gate = LocationAddress(0, 0, 0)
    result = _native.TargetSolver(engine, MoveSearch.entropy()).solve(
        {0: mem._inner}, {0: gate._inner}, [], None
    )
    assert result.status == "solved"
    assert result.proven is False
    assert result.stage_expansions == []
    assert result.plan_stage is None
