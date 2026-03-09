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


class TestWordNRows:
    """Verify Word.n_rows property works for different word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7, 10])
    def test_word_n_rows(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        for word in arch_spec.words:
            assert word.n_rows == word_size_y


class TestValidateInitialLayout:
    """Validate that initial layout validation respects dynamic n_rows."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_valid_layout(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        # All left-column sites (0..n_rows-1) should be valid
        valid_layout = tuple(
            LocationAddress(word_id, site_id)
            for word_id in range(2)
            for site_id in range(word_size_y)
        )
        # Should not raise
        methods.validate_initial_layout(valid_layout[:word_size_y])

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_invalid_right_column_site(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        # A site_id == n_rows is a right-column site, should be rejected
        invalid_layout = (
            LocationAddress(0, 0),
            LocationAddress(0, word_size_y),
        )
        with pytest.raises(ValueError, match="site ids"):
            methods.validate_initial_layout(invalid_layout)


class TestDesiredCzLayout:
    """Test CZ layout computation with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_same_word_cz(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        n = word_size_y
        # Two qubits on the same word, left column
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(0, 1)),
            move_count=(0, 0),
        )
        result = methods.desired_cz_layout(state, controls=(0,), targets=(1,))
        # One qubit should move to right column (site_id + n_rows)
        layouts = result.layout
        right_sites = [addr for addr in layouts if addr.site_id >= n]
        assert len(right_sites) == 1
        # The right-column site should be offset by n_rows from a left-column site
        for addr in right_sites:
            assert addr.site_id < 2 * n

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cross_word_cz(self, word_size_y: int):
        methods = _make_placement_methods(word_size_y)
        n = word_size_y
        # Two qubits on different words
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
            move_count=(0, 0),
        )
        result = methods.desired_cz_layout(state, controls=(0,), targets=(1,))
        # One qubit should end up on the right column
        right_sites = [addr for addr in result.layout if addr.site_id >= n]
        assert len(right_sites) == 1
        for addr in right_sites:
            assert addr.site_id < 2 * n


class TestComputeMoveLayers:
    """Test move synthesis with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_site_bus_move(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        n = word_size_y
        # Move a qubit from left to right column within same word
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
        # Verify all lanes in the result are valid
        for layer in layers:
            for lane in layer:
                assert arch_spec.validate_lane(lane) == set()

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cross_word_move(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        n = word_size_y
        # Move qubit from word 0 to word 1, ending on right column
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
    def test_return_from_right_column(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        n = word_size_y
        # One qubit on right column, one on left
        state_before = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, n), LocationAddress(0, 1)),
            move_count=(1, 0),
        )
        mid_state, left_move_layers = placement.choose_return_layout(
            state_before, controls=(0,), targets=(1,)
        )
        # After return, all qubits should be on left column
        for addr in mid_state.layout:
            assert addr.site_id < n, f"Qubit still on right column after return: {addr}"

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_left_sites_enumeration(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        left_sites = placement._left_sites()
        assert len(left_sites) == 2 * word_size_y  # 2 words * n_rows sites each
        for addr in left_sites:
            assert addr.site_id < word_size_y
            assert addr.word_id in (0, 1)

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_distance_key_uses_dynamic_offset(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        n = word_size_y
        right_addr = LocationAddress(0, n)  # first right-column site
        left_addr = LocationAddress(0, 0)
        key = placement._distance_key(right_addr, left_addr)
        # word_distance=0, site_distance=0, word_id=0, site_id=0
        assert key == (0, 0, 0, 0)

        right_addr = LocationAddress(0, n + 1)
        left_addr = LocationAddress(0, 0)
        key = placement._distance_key(right_addr, left_addr)
        # word_distance=0, site_distance=1 (row 1 vs site 0)
        assert key == (0, 1, 0, 0)


class TestFullCzPipeline:
    """End-to-end CZ placement with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cz_placements_end_to_end(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        n = word_size_y
        # Start with 2 qubits on left column of word 0
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(0, 1)),
            move_count=(0, 0),
        )
        result = placement.cz_placements(state, controls=(0,), targets=(1,))
        assert isinstance(result, ExecuteCZ)
        # One qubit should be on right column for CZ
        right_count = sum(1 for addr in result.layout if addr.site_id >= n)
        assert right_count == 1
        # Move layers should be non-empty
        assert len(result.get_move_layers()) > 0

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_cz_placements_cross_word(self, word_size_y: int):
        placement = _make_nohome(word_size_y)
        n = word_size_y
        # Qubits on different words
        state = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
            move_count=(0, 0),
        )
        result = placement.cz_placements(state, controls=(0,), targets=(1,))
        assert isinstance(result, ExecuteCZ)
        right_count = sum(1 for addr in result.layout if addr.site_id >= n)
        assert right_count == 1


class TestMoveToLeft:
    """Test move_to_left with varying word sizes."""

    @pytest.mark.parametrize("word_size_y", [3, 5, 7])
    def test_move_to_left_reverse(self, word_size_y: int):
        arch_spec = _make_arch(word_size_y)
        n = word_size_y
        state_before = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(1, n), LocationAddress(1, 0)),
            move_count=(1, 0),
        )
        state_after = ConcreteState(
            occupied=frozenset(),
            layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
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
