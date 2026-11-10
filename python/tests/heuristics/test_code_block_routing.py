"""Block-aware preferences choose real, architecture-valid CZ routes."""

from types import SimpleNamespace

import pytest

from bloqade.lanes.analysis.code_blocks import LocalCodeBlock
from bloqade.lanes.analysis.placement import (
    ConcreteState,
    ExecuteCZ,
    PalindromePlacementStrategy,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import MoverSelection, _native
from bloqade.lanes.bytecode.encoding import LocationAddress as L
from bloqade.lanes.heuristics.physical.code_block_routing import (
    coordinated_targets,
    inter_block_groups,
    valid_cz_goal,
)
from bloqade.lanes.heuristics.physical.movement import make_physical_placement_strategy
from bloqade.lanes.heuristics.physical.nohome import NoHomePlacementStrategy


def state(layout=None):
    return ConcreteState(
        occupied=frozenset(),
        layout=layout or (L(0, 0), L(0, 1), L(2, 0), L(2, 1)),
        move_count=(0, 0, 0, 0),
        code_blocks=(LocalCodeBlock(0, (0, 1)), LocalCodeBlock(1, (2, 3))),
    )


def strategy(*, candidates=8, return_moves=False, **kwargs):
    return make_physical_placement_strategy(
        arch_spec=get_arch_spec(),
        search_budget=500,
        code_block_candidates=candidates,
        return_moves=return_moves,
        **kwargs,
    )


def replay(arch, before, after, pairs=((0, 2), (3, 1))):
    """Check every emitted bus shot and replay all physical atoms, including spectators."""
    layout = list(before.layout)
    for layer in after.move_layers:
        assert not arch.check_lane_group(layer)
        endpoints = dict(arch.get_endpoints(lane) for lane in layer)
        assert not before.occupied.intersection(endpoints)
        layout = [endpoints.get(loc, loc) for loc in layout]
        assert len(set(layout)) == len(layout)
        assert not before.occupied.intersection(layout)
    assert tuple(layout) == after.layout
    assert valid_cz_goal(arch, before, list(pairs), dict(enumerate(layout)))


@pytest.mark.parametrize("return_moves", [False, True])
def test_coordinated_movers_improve_mixed_control_directions(return_moves):
    before = state()
    ordinary = strategy(
        candidates=0, return_moves=return_moves, mover_selection=MoverSelection.RULE
    )
    aware = strategy(return_moves=return_moves, mover_selection=MoverSelection.RULE)
    baseline = ordinary.cz_placements(before, (0, 3), (2, 1))
    result = aware.cz_placements(before, (0, 3), (2, 1))
    assert isinstance(baseline, ExecuteCZ) and isinstance(result, ExecuteCZ)
    assert len(result.move_layers) < len(baseline.move_layers)
    assert result.code_blocks == before.code_blocks
    replay(aware.arch_spec, before, result)
    if return_moves:
        assert isinstance(aware, PalindromePlacementStrategy)
        restored = aware._unwrap(result)
        assert isinstance(restored, ConcreteState)
        assert restored.layout == before.layout
        assert restored.code_blocks == before.code_blocks
        again = aware.cz_placements(result, (0, 3), (2, 1))
        assert isinstance(again, ExecuteCZ)
        assert again.move_layers == result.move_layers


def test_default_ranked_router_improves_irregular_blocks():
    before = state((L(0, 0), L(4, 1), L(2, 0), L(2, 2)))
    baseline = strategy(candidates=0).cz_placements(before, (0, 3), (2, 1))
    aware = strategy()
    result = aware.cz_placements(before, (0, 3), (2, 1))
    assert isinstance(baseline, ExecuteCZ) and isinstance(result, ExecuteCZ)
    assert len(result.move_layers) < len(baseline.move_layers)
    replay(aware.arch_spec, before, result)


def test_non_return_routing_uses_current_locations_on_next_layer():
    router = strategy(mover_selection=MoverSelection.RULE)
    before = state()
    first = router.cz_placements(before, (0, 3), (2, 1))
    assert isinstance(first, ExecuteCZ)
    assert first.layout != before.layout
    second = router.cz_placements(first, (0, 1), (3, 2))
    assert isinstance(second, ExecuteCZ)
    assert second.code_blocks == before.code_blocks
    replay(router.arch_spec, first, second, pairs=((0, 3), (1, 2)))


def test_only_inter_block_participants_are_grouped():
    before = ConcreteState(
        occupied=frozenset(),
        layout=tuple(L(0, i) for i in range(8)),
        move_count=(0,) * 8,
        code_blocks=(
            LocalCodeBlock(0, (0, 1, 2, 3, None)),
            LocalCodeBlock(1, (4, 5, 6, 7)),
        ),
    )
    _, groups = inter_block_groups(before, [(0, 4), (1, 5), (2, 3)])
    assert groups == ((0, 1), (4, 5))


@pytest.mark.parametrize("registered", [False, True])
def test_no_preference_for_unregistered_or_intra_block_layers(registered):
    before = state()
    if not registered:
        before.code_blocks = ()
    controls, targets = ((0, 2), (1, 3)) if registered else ((0, 3), (2, 1))
    ordinary, aware = strategy(candidates=0), strategy()
    a = ordinary.cz_placements(before, controls, targets)
    b = aware.cz_placements(before, controls, targets)
    assert isinstance(a, ExecuteCZ) and isinstance(b, ExecuteCZ)
    assert a.move_layers == b.move_layers
    assert a.layout == b.layout
    assert isinstance(ordinary, NoHomePlacementStrategy)
    assert isinstance(aware, NoHomePlacementStrategy)
    assert ordinary.rust_nodes_expanded_total == aware.rust_nodes_expanded_total


def test_invalid_preferred_targets_fall_back(monkeypatch):
    from bloqade.lanes.heuristics.physical import code_block_routing

    before = state()
    # Every atom on one site: preference must never bypass collision checks.
    monkeypatch.setattr(
        code_block_routing,
        "coordinated_targets",
        lambda *args: [{q: L(0, 0) for q in range(4)}],
    )
    ordinary = strategy(candidates=0, mover_selection=MoverSelection.RULE)
    aware = strategy(mover_selection=MoverSelection.RULE)
    a = ordinary.cz_placements(before, (0, 3), (2, 1))
    b = aware.cz_placements(before, (0, 3), (2, 1))
    assert isinstance(a, ExecuteCZ) and isinstance(b, ExecuteCZ)
    assert a.move_layers == b.move_layers


def test_spectator_pairs_are_not_valid_targets():
    before = state()
    goal = {0: L(0, 0), 1: L(2, 0), 2: L(1, 0), 3: L(3, 0)}
    assert not valid_cz_goal(get_arch_spec(), before, [(0, 2)], goal)
    assert valid_cz_goal(get_arch_spec(), before, [(0, 2), (1, 3)], goal)


def test_candidate_generation_preserves_other_pairs_and_spectators():
    before = ConcreteState(
        occupied=frozenset(),
        layout=(L(0, 0), L(0, 1), L(2, 0), L(2, 1), L(4, 0), L(4, 1), L(6, 0)),
        move_count=(0,) * 7,
        code_blocks=(
            LocalCodeBlock(0, (0, 1)),
            LocalCodeBlock(1, (2, 3)),
            LocalCodeBlock(2, (4, 5, 6)),
        ),
    )
    goal = dict(enumerate(before.layout))
    goal[5] = L(5, 0)
    candidates = coordinated_targets(
        get_arch_spec(), before, [(0, 2), (3, 1), (4, 5)], goal, 8
    )
    assert candidates
    assert all(all(c[q] == goal[q] for q in (4, 5, 6)) for c in candidates)


def test_candidate_cap_and_invalid_knob():
    before = state()
    assert (
        len(
            coordinated_targets(
                get_arch_spec(),
                before,
                [(0, 2), (3, 1)],
                dict(enumerate(before.layout)),
                1,
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="code_block_candidates"):
        strategy(candidates=-1)


def test_bus_capacity_remains_a_hard_limit():
    base = get_arch_spec()
    arch = ArchSpec.from_components(
        words=base.words,
        zones=base.zones,
        modes=base.modes,
        zone_buses=base.zone_buses,
        paths=dict(base.paths),
        aod_capacity=(1, 1),
    )
    before = state()
    router = make_physical_placement_strategy(
        arch_spec=arch,
        return_moves=False,
        search_budget=1000,
        mover_selection=MoverSelection.RULE,
    )
    result = router.cz_placements(before, (0, 3), (2, 1))
    assert isinstance(result, ExecuteCZ)
    replay(arch, before, result)
    for layer in result.move_layers:
        positions = [arch.get_position(arch.get_endpoints(lane)[0]) for lane in layer]
        assert len({x for x, _ in positions}) <= 1
        assert len({y for _, y in positions}) <= 1


def test_odd_block_cycle_keeps_ordinary_targets():
    before = ConcreteState(
        occupied=frozenset(),
        layout=(L(0, 0), L(0, 1), L(2, 0), L(2, 1), L(4, 0), L(4, 1)),
        move_count=(0,) * 6,
        code_blocks=(
            LocalCodeBlock(0, (0, 1)),
            LocalCodeBlock(1, (2, 3)),
            LocalCodeBlock(2, (4, 5)),
        ),
    )
    assert (
        coordinated_targets(
            get_arch_spec(),
            before,
            [(0, 2), (3, 4), (5, 1)],
            dict(enumerate(before.layout)),
            8,
        )
        == []
    )


def test_preference_budget_failure_preserves_ordinary_route(monkeypatch):
    before = state()
    ordinary = strategy(candidates=0, mover_selection=MoverSelection.RULE)
    baseline = ordinary.cz_placements(before, (0, 3), (2, 1))
    budgets = []

    class BudgetSolver:
        def __init__(self, *args):
            pass

        def solve(self, *args, max_expansions):
            budgets.append(max_expansions)
            return SimpleNamespace(
                status=_native.SolveStatus.BUDGET_EXCEEDED,
                nodes_expanded=max_expansions,
            )

    monkeypatch.setattr(_native, "TargetSolver", BudgetSolver)
    aware = strategy(mover_selection=MoverSelection.RULE)
    result = aware.cz_placements(before, (0, 3), (2, 1))
    assert isinstance(baseline, ExecuteCZ) and isinstance(result, ExecuteCZ)
    assert result.move_layers == baseline.move_layers
    assert result.layout == baseline.layout
    assert budgets == [500]
    assert isinstance(aware, NoHomePlacementStrategy)
    assert isinstance(ordinary, NoHomePlacementStrategy)
    assert aware.rust_nodes_expanded_total == ordinary.rust_nodes_expanded_total + 500


def test_physical_pipeline_passes_membership_to_successive_routes(monkeypatch):
    from kirin.dialects import ilist

    from bloqade import squin
    from bloqade.gemini.physical import kernel, qubit
    from bloqade.lanes.heuristics.physical._no_return_base import NoReturnStrategyBase
    from bloqade.lanes.transform import PhysicalPipeline

    @kernel
    def program():
        blocks = qubit.qalloc_at(ilist.IList([0, 1]))
        squin.broadcast.cz(
            ilist.IList([blocks[0][0], blocks[1][1]]),
            ilist.IList([blocks[1][0], blocks[0][1]]),
        )
        squin.x(blocks[0][0])
        squin.broadcast.cz(blocks[0], blocks[1])
        return squin.broadcast.measure(blocks[0] + blocks[1])

    seen = []
    original = NoReturnStrategyBase._prefer_code_block_route

    def record(self, state, pairs, ordinary):
        seen.append(inter_block_groups(state, pairs)[1])
        return original(self, state, pairs, ordinary)

    monkeypatch.setattr(NoReturnStrategyBase, "_prefer_code_block_route", record)
    PhysicalPipeline(slots=[[(0, 0, 0), (0, 0, 4)], [(0, 0, 2), (0, 0, 6)]]).emit(
        program, no_raise=False
    )
    assert len(seen) == 2
    assert all(sorted(map(len, groups)) == [2, 2] for groups in seen)
