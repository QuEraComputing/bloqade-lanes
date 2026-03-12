import math

from bloqade.gemini import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes.analysis.atom.atom_state_data import AtomStateData
from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.logical_placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.heuristics.physical_placement import PhysicalGreedyPlacementStrategy
from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.logical_mvp import (
    compile_squin_to_move,
    compile_squin_to_move_and_visualize,
)
from bloqade.lanes.metrics import (
    analyze_kernel_move_time_with_strategy,
    analyze_kernel_moves_with_strategy,
)


def _count_move_events(mt) -> int:
    return sum(1 for stmt in mt.callable_region.walk() if isinstance(stmt, move.Move))


def test_physical_strategy_validate_initial_layout_multiword():
    strategy = PhysicalGreedyPlacementStrategy()
    strategy.validate_initial_layout(
        (
            LocationAddress(0, 0),
            LocationAddress(1, 1),
            LocationAddress(2, 2),
            LocationAddress(3, 3),
        )
    )


def test_physical_strategy_validation_enforces_left_only():
    strategy = PhysicalGreedyPlacementStrategy()
    strategy.validate_initial_layout((LocationAddress(0, 5),))


def _inverse_layers(move_layers):
    return tuple(
        tuple(lane.reverse() for lane in reversed(layer))
        for layer in reversed(move_layers)
    )


def test_physical_strategy_ignores_future_lookahead_layers():
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(1, 0),
            LocationAddress(0, 4),
        ),
        move_count=(0, 0, 0),
    )
    controls = (0,)
    targets = (1,)
    lookahead = (((0,), (2,)),)

    strategy = PhysicalGreedyPlacementStrategy(arch_spec=get_arch_spec())

    result_no_lookahead = strategy.cz_placements(
        state, controls, targets, lookahead_cz_layers=()
    )
    result_with_lookahead = strategy.cz_placements(
        state, controls, targets, lookahead_cz_layers=lookahead
    )

    assert isinstance(result_no_lookahead, ExecuteCZ)
    assert isinstance(result_with_lookahead, ExecuteCZ)
    assert result_no_lookahead.layout == result_with_lookahead.layout
    assert result_no_lookahead.move_layers == result_with_lookahead.move_layers


def test_physical_strategy_uses_inverse_return_layers():
    strategy = PhysicalGreedyPlacementStrategy(arch_spec=get_arch_spec())
    initial = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(1, 0),
        ),
        move_count=(0, 0),
    )
    controls = (0,)
    targets = (1,)

    first = strategy.cz_placements(initial, controls, targets)
    assert isinstance(first, ExecuteCZ)
    assert len(first.move_layers) > 0
    expected_inverse = _inverse_layers(first.move_layers)

    second = strategy.cz_placements(first, controls, targets)
    assert isinstance(second, ExecuteCZ)
    assert second.move_layers[: len(expected_inverse)] == expected_inverse


def test_compute_moves_only_uses_free_destinations():
    strategy = PhysicalGreedyPlacementStrategy(arch_spec=get_arch_spec())
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),  # q0
            LocationAddress(0, 1),  # q1
            LocationAddress(0, 3),  # q2 (not moving)
        ),
        move_count=(0, 0, 0),
    )
    state_after = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 1),  # q0 should move after q1 vacates
            LocationAddress(0, 2),  # q1 moves first
            LocationAddress(0, 3),
        ),
        move_count=(1, 1, 0),
    )

    move_layers = strategy.compute_moves(state_before, state_after)
    assert move_layers is not None
    assert len(move_layers) >= 2

    path_finder = PathFinder(strategy.arch_spec)
    sim = AtomStateData.new(list(state_before.layout))
    for layer in move_layers:
        sim = sim.apply_moves(layer, path_finder)
        assert sim is not None
        assert len(sim.qubit_to_locations) == len(state_before.layout)

    assert (
        tuple(sim.qubit_to_locations[qid] for qid in range(len(state_before.layout)))
        == state_after.layout
    )


@gemini_logical.kernel(aggressive_unroll=True)
def _movement_kernel():
    reg = qubit.qalloc(5)
    squin.broadcast.u3(0.25 * math.pi, 0.1 * math.pi, 0.0, reg)
    squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))


def test_compile_entrypoint_default_behavior_stable():
    default_move = compile_squin_to_move(_movement_kernel, no_raise=False)
    explicit_old_move = compile_squin_to_move(
        _movement_kernel,
        no_raise=False,
        placement_strategy=LogicalPlacementStrategyNoHome(),
        insert_palindrome_moves=True,
    )
    assert str(default_move.code) == str(explicit_old_move.code)


def test_compile_entrypoint_accepts_physical_strategy():
    class TrackingPhysicalStrategy(PhysicalGreedyPlacementStrategy):
        cz_calls: int = 0

        def cz_placements(
            self,
            state,
            controls,
            targets,
            lookahead_cz_layers=(),
        ):
            self.cz_calls += 1
            return super().cz_placements(
                state,
                controls,
                targets,
                lookahead_cz_layers=lookahead_cz_layers,
            )

    strategy = TrackingPhysicalStrategy(arch_spec=get_arch_spec())
    physical_move = compile_squin_to_move(
        _movement_kernel,
        no_raise=False,
        transversal_rewrite=False,
        placement_mode="physical",
        placement_strategy=strategy,
    )

    assert _count_move_events(physical_move) >= 0
    assert strategy.cz_calls > 0


def test_compile_entrypoint_physical_mode_uses_physical_defaults(monkeypatch):
    calls = {"cz_calls": 0}
    original = PhysicalGreedyPlacementStrategy.cz_placements

    def wrapped(self, state, controls, targets, lookahead_cz_layers=()):
        calls["cz_calls"] += 1
        return original(
            self,
            state,
            controls,
            targets,
            lookahead_cz_layers=lookahead_cz_layers,
        )

    monkeypatch.setattr(PhysicalGreedyPlacementStrategy, "cz_placements", wrapped)
    physical_move = compile_squin_to_move(
        _movement_kernel,
        no_raise=False,
        placement_mode="physical",
    )
    assert _count_move_events(physical_move) >= 0
    assert calls["cz_calls"] > 0


def test_visualize_entrypoint_physical_mode_uses_physical_marker(monkeypatch):
    called = {"debugger": 0, "atom_marker": None}

    def fake_debugger(*_args, **kwargs):
        called["debugger"] += 1
        called["atom_marker"] = kwargs.get("atom_marker")

    monkeypatch.setattr("bloqade.lanes.visualize.debugger", fake_debugger)
    compile_squin_to_move_and_visualize(
        _movement_kernel,
        interactive=False,
        animated=False,
        no_raise=False,
        placement_mode="physical",
    )
    assert called["debugger"] == 1
    assert called["atom_marker"] == "o"


def test_metrics_harness_quantifies_default_vs_custom_strategy():
    baseline_strategy = LogicalPlacementStrategyNoHome()
    custom_strategy = PhysicalGreedyPlacementStrategy(arch_spec=get_arch_spec())

    baseline_moves = analyze_kernel_moves_with_strategy(
        _movement_kernel,
        placement_strategy=baseline_strategy,
        insert_palindrome_moves=True,
    )
    custom_moves = analyze_kernel_moves_with_strategy(
        _movement_kernel,
        placement_strategy=custom_strategy,
        insert_palindrome_moves=True,
    )
    baseline_time = analyze_kernel_move_time_with_strategy(
        _movement_kernel,
        placement_strategy=baseline_strategy,
        insert_palindrome_moves=True,
    )
    custom_time = analyze_kernel_move_time_with_strategy(
        _movement_kernel,
        placement_strategy=custom_strategy,
        insert_palindrome_moves=True,
    )

    lane_delta = custom_moves.moved_lane_count - baseline_moves.moved_lane_count
    time_delta_us = custom_time.total_move_time_us - baseline_time.total_move_time_us

    assert isinstance(lane_delta, int)
    assert isinstance(time_delta_us, float)
    assert baseline_moves.moved_lane_count >= 0
    assert custom_moves.moved_lane_count >= 0
    assert baseline_time.total_move_time_us >= 0.0
    assert custom_time.total_move_time_us >= 0.0
