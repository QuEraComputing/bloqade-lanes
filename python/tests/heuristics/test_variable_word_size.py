"""Tests for heuristics with varying word sizes (n_rows != 5)."""

import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.analysis.placement.lattice import ExecuteCZ
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.heuristics.logical_placement import (
    LogicalPlacementMethods,
    LogicalPlacementStrategyNoHome,
)
from bloqade.lanes.heuristics.move_synthesis import compute_move_layers, move_to_left
from bloqade.lanes.layout.encoding import LocationAddress


def _make_arch(word_size_y: int) -> layout.ArchSpec:
    return generate_arch_hypercube(hypercube_dims=1, word_size_y=word_size_y)


def _make_placement_methods(word_size_y: int) -> LogicalPlacementMethods:
    arch_spec = _make_arch(word_size_y)
    methods = LogicalPlacementMethods(arch_spec=arch_spec)
    return methods


def _make_nohome(word_size_y: int) -> LogicalPlacementStrategyNoHome:
    arch_spec = _make_arch(word_size_y)
    placement = LogicalPlacementStrategyNoHome.__new__(LogicalPlacementStrategyNoHome)
    placement.arch_spec = arch_spec
    placement.H_lookahead = 4
    placement.gamma = 0.85
    placement.lambda_lookahead = 0.5
    placement.K_candidates = 8
    placement.large_cost = 1e9
    placement.lane_move_overhead_cost = 0.0
    placement.top_bus_signatures = 6
    placement.bus_reward_rho = 0.7
    placement._best_path_cache = {}
    placement.__post_init__()
    return placement


# ── Shared validation helpers ──


def assert_valid_cz_placement(
    arch_spec: layout.ArchSpec,
    result: ExecuteCZ,
    controls: tuple[int, ...],
    targets: tuple[int, ...],
) -> None:
    """Assert CZ placement result has valid blockade pairs and lanes."""
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


def assert_all_home(arch_spec: layout.ArchSpec, addrs: tuple[LocationAddress, ...]) -> None:
    """Assert all addresses are at home positions."""
    for addr in addrs:
        assert arch_spec.is_home_position(addr), f"{addr} is not a home position"


# ── Tests ──


class TestWordNRows:
    """Verify Word.n_rows property works for different word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7, 10])
    def test_word_n_rows(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        for word in arch_spec.words:
            assert word.n_rows == word_size_y


class TestValidateInitialLayout:
    """Validate that initial layout validation respects home positions."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_valid_layout(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        arch = methods.arch_spec
        # All home word sites should be valid
        valid_layout = tuple(
            LocationAddress(word_id, site_id)
            for word_id in arch._home_words
            for site_id in range(len(arch.words[word_id].site_indices))
        )
        # Should not raise (use a subset)
        methods.validate_initial_layout(valid_layout[:word_size_y])

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_invalid_non_home_position(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        arch = methods.arch_spec
        # Find a non-home word
        non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
        assert len(non_home) > 0, "Need at least one non-home word"
        invalid_layout = (
            LocationAddress(0, 0),
            LocationAddress(non_home[0], 0),
        )
        with pytest.raises(ValueError, match="not at a home position"):
            methods.validate_initial_layout(invalid_layout)


class TestDesiredCzLayout:
    """Test CZ layout computation with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_same_word_cz(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        arch = methods.arch_spec
        # Two qubits on the same home word
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(0, 1)),
            move_count=(0, 0),
        )
        result = methods.desired_cz_layout(state, controls=(0,), targets=(1,))
        assert_valid_cz_placement(arch, result, controls=(0,), targets=(1,))

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cross_word_cz(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        arch = methods.arch_spec
        # Two qubits on different home words (both at site 0)
        # Word 0 ↔ Word 1: already at blockade positions, may need 0 moves
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
            move_count=(0, 0),
        )
        result = methods.desired_cz_layout(state, controls=(0,), targets=(1,))
        assert_valid_cz_placement(arch, result, controls=(0,), targets=(1,))


class TestComputeMoveLayers:
    """Test move synthesis with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_same_word_move(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        n = word_size_y
        # Move a qubit within the same word (left col → right col via site bus)
        state_before = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
            move_count=(0, 0),
        )
        state_after = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, n), LocationAddress(1, 0)),
            move_count=(1, 0),
        )
        layers = compute_move_layers(arch_spec, state_before, state_after)
        assert len(layers) > 0
        for layer in layers:
            for lane in layer:
                assert arch_spec.validate_lane(lane) == set()

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cross_word_move(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        n = word_size_y
        # Move qubit from word 0 to word 1: site bus to right col, then word bus
        state_before = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
            move_count=(0, 0),
        )
        state_after = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(1, n), LocationAddress(1, 0)),
            move_count=(1, 0),
        )
        layers = compute_move_layers(arch_spec, state_before, state_after)
        assert len(layers) > 0
        for layer in layers:
            for lane in layer:
                assert arch_spec.validate_lane(lane) == set()

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_no_diff_no_moves(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
            move_count=(0, 0),
        )
        layers = compute_move_layers(arch_spec, state, state)
        assert layers == ()


class TestNoHomeReturnLayout:
    """Test LogicalPlacementStrategyNoHome with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_return_from_non_home(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        arch = placement.arch_spec
        # One qubit at non-home (CZ-staging) word, one at home
        non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
        assert len(non_home) > 0
        state_before = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(non_home[0], 0), LocationAddress(0, 1)),
            move_count=(1, 0),
        )
        mid_state, _ = placement.choose_return_layout(
            state_before, controls=(0,), targets=(1,)
        )
        assert_all_home(arch, mid_state.layout)

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_home_sites_enumeration(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        arch = placement.arch_spec
        home_sites = placement._home_sites()
        # Home word(s) × all sites per word
        expected_count = sum(
            len(arch.words[w].site_indices) for w in arch._home_words
        )
        assert len(home_sites) == expected_count
        for addr in home_sites:
            assert arch.is_home_position(addr)

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_distance_key_same_word(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        arch = placement.arch_spec
        non_home = [w for w in range(len(arch.words)) if w not in arch._home_words]
        assert len(non_home) > 0
        # CZ address in non-home word, home address in word 0
        cz_addr = LocationAddress(non_home[0], 0)
        home_addr = LocationAddress(0, 0)
        key = placement._distance_key(cz_addr, home_addr)
        # blockade partner of (non_home, 0) is (0, 0) — same as home_addr
        # word_distance=0, site_distance=0
        assert key[0] == 0  # word_distance
        assert key[1] == 0  # site_distance


class TestFullCzPipeline:
    """End-to-end CZ placement with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cz_placements_end_to_end(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        arch = placement.arch_spec
        # Start with 2 qubits on home word
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(0, 1)),
            move_count=(0, 0),
        )
        result = placement.cz_placements(state, controls=(0,), targets=(1,))
        assert isinstance(result, ExecuteCZ)
        assert_valid_cz_placement(arch, result, controls=(0,), targets=(1,))
        assert len(result.get_move_layers()) > 0

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cz_placements_cross_word(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        arch = placement.arch_spec
        # Qubits at matching sites in paired words — already CZ-ready
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
            move_count=(0, 0),
        )
        result = placement.cz_placements(state, controls=(0,), targets=(1,))
        assert isinstance(result, ExecuteCZ)
        assert_valid_cz_placement(arch, result, controls=(0,), targets=(1,))


class TestMoveToLeft:
    """Test move_to_left with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_move_to_left_reverse(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        # Move qubit from non-home word back to home
        state_before = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(1, 0), LocationAddress(0, 1)),
            move_count=(1, 0),
        )
        state_after = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(0, 1)),
            move_count=(2, 0),
        )
        out_state, layers = move_to_left(arch_spec, state_before, state_after)
        assert out_state == state_after
        # Should produce reverse of forward layers
        forward_layers = compute_move_layers(arch_spec, state_after, state_before)
        expected_layers = tuple(
            tuple(lane.reverse() for lane in move_lanes[::-1])
            for move_lanes in forward_layers[::-1]
        )
        assert layers == expected_layers
