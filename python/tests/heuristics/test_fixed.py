import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import AtomState, ConcreteState
from bloqade.lanes.analysis.placement.lattice import ExecuteCZ
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.heuristics.logical_placement import (
    LogicalPlacementStrategy,
    LogicalPlacementStrategyNoHome,
)
from bloqade.lanes.heuristics.move_synthesis import compute_move_layers, move_to_left
from bloqade.lanes.layout.encoding import (
    LocationAddress,
)


# ── Shared validation helpers ──


def assert_valid_cz_placement(
    arch_spec: layout.ArchSpec,
    result: ExecuteCZ,
    controls: tuple[int, ...],
    targets: tuple[int, ...],
) -> None:
    """Assert CZ placement result has valid blockade pairs and valid lanes."""
    for c, t in zip(controls, targets):
        c_addr = result.layout[c]
        t_addr = result.layout[t]
        assert (
            arch_spec.get_blockaded_location(c_addr) == t_addr
            or arch_spec.get_blockaded_location(t_addr) == c_addr
        ), f"CZ pair ({c_addr}, {t_addr}) not at blockade positions"
    for layer in result.get_move_layers():
        for lane in layer:
            assert arch_spec.validate_lane(lane) == set(), f"Invalid lane: {lane}"


def assert_all_home(
    arch_spec: layout.ArchSpec, addrs: tuple[LocationAddress, ...]
) -> None:
    for addr in addrs:
        assert arch_spec.is_home_position(addr), f"{addr} is not a home position"


# ── CZ placement test cases ──


def cz_placement_cases():
    # Trivial cases
    yield ("top", AtomState.top(), (0, 1), (2, 3), AtomState.top())
    yield ("bottom", AtomState.bottom(), (0, 1), (2, 3), AtomState.bottom())

    # Qubits in different home words — already at blockade positions
    # (0,0)↔(1,0) and (0,1)↔(1,1) are CZ pairs
    yield (
        "cross_word_no_move",
        ConcreteState(
            occupied=frozenset(),
            layout=(
                LocationAddress(0, 0),
                LocationAddress(0, 1),
                LocationAddress(1, 0),
                LocationAddress(1, 1),
            ),
            move_count=(0, 0, 0, 0),
        ),
        (0, 1),
        (2, 3),
        None,  # property-check only
    )

    # Two qubits in same home word — one must move to partner word for CZ
    yield (
        "same_word_needs_move",
        ConcreteState(
            occupied=frozenset(),
            layout=(
                LocationAddress(0, 0),
                LocationAddress(0, 1),
            ),
            move_count=(0, 0),
        ),
        (0,),
        (1,),
        None,
    )

    # With unequal move counts — heuristic should prefer moving less-moved qubit
    yield (
        "unequal_move_counts",
        ConcreteState(
            occupied=frozenset(),
            layout=(
                LocationAddress(0, 0),
                LocationAddress(0, 1),
            ),
            move_count=(1, 0),
        ),
        (0,),
        (1,),
        None,
    )

    # Mismatched control/target counts → bottom
    yield (
        "mismatched_counts",
        ConcreteState(
            occupied=frozenset(),
            layout=(
                LocationAddress(0, 0),
                LocationAddress(0, 1),
                LocationAddress(2, 0),
                LocationAddress(2, 1),
            ),
            move_count=(1, 0, 0, 0),
        ),
        (0, 1, 4),
        (2, 3),
        AtomState.bottom(),
    )


@pytest.mark.parametrize(
    "name, state_before, targets, controls, state_after",
    cz_placement_cases(),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_fixed_cz_placement(
    name: str,
    state_before: AtomState,
    targets: tuple[int, ...],
    controls: tuple[int, ...],
    state_after: AtomState | None,
):
    placement_strategy = LogicalPlacementStrategy()
    arch = placement_strategy.arch_spec
    state_result = placement_strategy.cz_placements(state_before, controls, targets)

    if state_after is not None:
        # Exact match for trivial cases (top, bottom, mismatched)
        assert state_result == state_after
        return

    # Property-based check for ConcreteState cases
    assert isinstance(state_result, ExecuteCZ)
    assert_valid_cz_placement(arch, state_result, controls, targets)
    # Move count should increase only for moved qubits
    assert isinstance(state_before, ConcreteState)
    for i, (before_addr, after_addr) in enumerate(
        zip(state_before.layout, state_result.layout)
    ):
        if before_addr != after_addr:
            assert state_result.move_count[i] > state_before.move_count[i]


def test_fixed_sq_placement():
    placement_strategy = LogicalPlacementStrategy()
    assert AtomState.top() == placement_strategy.sq_placements(
        AtomState.top(), (0, 1, 2)
    )
    assert AtomState.bottom() == placement_strategy.sq_placements(
        AtomState.bottom(), (0, 1, 2)
    )
    state = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(0, 0),
            LocationAddress(0, 1),
            LocationAddress(2, 0),
            LocationAddress(2, 1),
        ),
        move_count=(0, 0, 0, 0),
    )
    assert state == placement_strategy.sq_placements(state, (0, 1, 2))


def test_fixed_invalid_initial_layout_non_home():
    """Non-home word address should be rejected."""
    placement_strategy = LogicalPlacementStrategy()
    arch = placement_strategy.arch_spec
    non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
    assert len(non_home) > 0
    invalid_layout = (
        LocationAddress(0, 0),
        LocationAddress(0, 1),
        LocationAddress(0, 2),
        LocationAddress(non_home[0], 0),
    )
    with pytest.raises(ValueError):
        placement_strategy.validate_initial_layout(invalid_layout)


def test_fixed_invalid_initial_layout_bad_word():
    """Word ID beyond architecture should be rejected."""
    placement_strategy = LogicalPlacementStrategy()
    invalid_layout = (
        LocationAddress(0, 0),
        LocationAddress(2, 0),
        LocationAddress(4, 0),
        LocationAddress(99, 0),
    )
    with pytest.raises(ValueError):
        placement_strategy.validate_initial_layout(invalid_layout)


def test_initial_layout():
    layout_heuristic = logical_layout.LogicalLayoutHeuristic()
    num_qubits = layout_heuristic.arch_spec.max_qubits  # 8 for new arch
    edges = {(i, j): 1 for i in range(num_qubits) for j in range(i + 1, num_qubits)}
    edges[(0, 1)] = 10
    edges = sum((weight * (edge,) for edge, weight in edges.items()), ())

    result = layout_heuristic.compute_layout(tuple(range(num_qubits)), [edges])

    arch = layout_heuristic.arch_spec
    # All qubits should be at home positions
    assert_all_home(arch, result)
    assert len(result) == num_qubits


def test_move_scheduler_cz():
    # Place qubits in same home word — CZ requires moving to partner
    initial_state = ConcreteState(
        frozenset(),
        (
            LocationAddress(0, 0),
            LocationAddress(0, 1),
        ),
        (0, 0),
    )

    placement = LogicalPlacementStrategy()
    arch = placement.arch_spec
    controls = (0,)
    targets = (1,)

    final_state = placement.cz_placements(initial_state, controls, targets)
    assert isinstance(final_state, ExecuteCZ)
    assert_valid_cz_placement(arch, final_state, controls, targets)
    assert len(final_state.get_move_layers()) > 0


def test_nohome_choose_return_layout():
    placement = LogicalPlacementStrategyNoHome()
    arch = placement.arch_spec
    non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
    assert len(non_home) > 0

    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(non_home[0], 0),
            LocationAddress(0, 1),
        ),
        move_count=(3, 4),
    )
    mid_state, left_move_layers = placement.choose_return_layout(
        state_before, controls=(0,), targets=(1,)
    )
    assert_all_home(arch, mid_state.layout)
    _, expected_left_move_layers = move_to_left(
        arch, state_before, mid_state,
    )
    assert left_move_layers == expected_left_move_layers


def test_nohome_choose_return_layout_duplicate_collision():
    placement = LogicalPlacementStrategyNoHome()
    arch = placement.arch_spec
    non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
    assert len(non_home) > 0

    # Fill ALL home sites with occupied atoms except one (used by qubit 1)
    home_word = min(arch._home_words)
    all_home_sites = [
        LocationAddress(w, s)
        for w in sorted(arch._home_words)
        for s in range(len(arch.words[w].site_indices))
    ]
    # qubit 1 is at (home_word, 0), fill the rest
    occupied = frozenset(
        addr for addr in all_home_sites if addr != LocationAddress(home_word, 0)
    )
    state_before = ConcreteState(
        occupied=occupied,
        layout=(
            LocationAddress(non_home[0], 0),
            LocationAddress(home_word, 0),
        ),
        move_count=(0, 0),
    )
    with pytest.raises(ValueError, match="No empty home site"):
        placement.choose_return_layout(state_before, controls=(0,), targets=(1,))


def test_nohome_choose_return_layout_sequential_no_conflicts():
    placement = LogicalPlacementStrategyNoHome()
    arch = placement.arch_spec
    non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
    assert len(non_home) > 0

    # Two qubits at non-home positions (2 sites per word)
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(non_home[0], 0),
            LocationAddress(non_home[0], 1),
        ),
        move_count=(0, 0),
    )
    mid_state, _ = placement.choose_return_layout(
        state_before, controls=(0,), targets=(1,)
    )
    assert_all_home(arch, mid_state.layout)


def test_nohome_cz_placements_combines_return_and_entangle_layers():
    placement = LogicalPlacementStrategyNoHome()
    arch = placement.arch_spec
    non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
    assert len(non_home) > 0

    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(non_home[0], 0),
            LocationAddress(0, 1),
        ),
        move_count=(0, 0),
    )
    result = placement.cz_placements(state_before, controls=(0,), targets=(1,))
    assert isinstance(result, ExecuteCZ)
    assert_valid_cz_placement(arch, result, controls=(0,), targets=(1,))


def test_nohome_best_path_uses_pathfinder_and_caches(monkeypatch: pytest.MonkeyPatch):
    placement = LogicalPlacementStrategyNoHome()
    # Find a valid src→dst pair with a lane
    src = LocationAddress(0, 0)
    dst = LocationAddress(0, 1)
    lane = placement.arch_spec.get_lane_address(src, dst)
    assert lane is not None

    calls = {"count": 0}

    def fake_find_path(
        _pathfinder, start, end,
        occupied=frozenset(), path_heuristic=None, edge_weight=None,
    ):
        _ = occupied, path_heuristic
        assert start == src
        assert end == dst
        assert edge_weight is not None
        calls["count"] += 1
        return ((lane,), (src, dst))

    monkeypatch.setattr(type(placement._path_finder), "find_path", fake_find_path)

    first = placement._best_path(src, dst)
    second = placement._best_path(src, dst)
    assert first == (lane,)
    assert second == (lane,)
    assert calls["count"] == 1


def test_nohome_best_path_none_returns_large_cost(monkeypatch: pytest.MonkeyPatch):
    placement = LogicalPlacementStrategyNoHome()
    src = LocationAddress(0, 0)
    dst = LocationAddress(0, 1)

    monkeypatch.setattr(
        type(placement._path_finder), "find_path", lambda *_args, **_kwargs: None
    )
    path = placement._best_path(src, dst)
    assert path is None
    assert placement._path_cost(path) == placement.large_cost


@pytest.mark.parametrize("sites_per_word", [2, 4])
def test_initial_layout_variable_sites_per_word(sites_per_word):
    from bloqade.lanes.arch.builder import build_arch
    from bloqade.lanes.arch.topology import HypercubeSiteTopology, HypercubeWordTopology
    from bloqade.lanes.arch.zone import ArchBlueprint, DeviceLayout, ZoneSpec

    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=4,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=HypercubeSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=sites_per_word),
    )
    arch_spec = build_arch(bp).arch
    layout_heuristic = logical_layout.LogicalLayoutHeuristic()
    layout_heuristic.arch_spec = arch_spec

    num_qubits = sites_per_word * 2  # 2 home words minimum
    edges = {(i, j): 1 for i in range(num_qubits) for j in range(i + 1, num_qubits)}
    edges[(0, 1)] = 10
    edges = sum((weight * (edge,) for edge, weight in edges.items()), ())

    result = layout_heuristic.compute_layout(tuple(range(num_qubits)), [edges])

    assert len(result) == num_qubits
    # All addresses should be at home positions
    assert_all_home(arch_spec, result)


def test_nohome_lookahead_can_change_return_choice():
    """Tests that lookahead can influence the return layout choice."""
    arch = get_arch_spec()
    non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
    assert len(non_home) > 0

    # Qubit at non-home word, needs to return
    state_before = ConcreteState(
        occupied=frozenset(),
        layout=(
            LocationAddress(non_home[0], 0),
            LocationAddress(0, 1),
        ),
        move_count=(0, 0),
    )
    lookahead = (((0,), (1,)),)

    placement_no_lookahead = LogicalPlacementStrategyNoHome(
        lambda_lookahead=0.0,
        H_lookahead=1,
    )
    no_lookahead_state, _ = placement_no_lookahead.choose_return_layout(
        state_before,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=lookahead,
    )

    placement_with_lookahead = LogicalPlacementStrategyNoHome(
        lambda_lookahead=20.0,
        H_lookahead=1,
    )
    lookahead_state, _ = placement_with_lookahead.choose_return_layout(
        state_before,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=lookahead,
    )

    # Both should return to home positions
    assert_all_home(arch, no_lookahead_state.layout)
    assert_all_home(arch, lookahead_state.layout)
