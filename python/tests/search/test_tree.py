"""Tests for ConfigurationTree and ExhaustiveMoveGenerator."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress, SiteLaneAddress
from bloqade.lanes.search.generators import ExhaustiveMoveGenerator
from bloqade.lanes.search.tree import ConfigurationTree


def _make_tree() -> ConfigurationTree:
    """Create a tree with the logical Gemini arch spec."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    return ConfigurationTree.from_initial_placement(arch_spec, placement)


def test_from_initial_placement():
    tree = _make_tree()
    assert tree.root.depth == 0
    assert tree.root.parent is None
    assert len(tree.root.configuration) == 2
    assert tree.root.config_key in tree.seen


def test_enumerate_single_moves():
    tree = _make_tree()
    moves = tree._enumerate_single_moves(tree.root)

    # Should find moves for both atoms
    assert len(moves) > 0

    # All sources should be occupied locations
    for lane, src, dst in moves:
        assert tree.root.is_occupied(src)
        assert not tree.root.is_occupied(dst)


def test_apply_move_set_valid():
    tree = _make_tree()

    # Site bus 0 forward: site 0 → site 5
    lane = SiteLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    child = tree._apply_move_set(tree.root, move_set)
    assert child is not None
    assert child.depth == 1
    assert child.parent is tree.root
    assert child.parent_moves == move_set
    # Qubit 0 should have moved from (0,0) to (0,5)
    assert child.configuration[0] == LocationAddress(0, 5)
    # Qubit 1 should be unchanged
    assert child.configuration[1] == LocationAddress(1, 0)


def test_apply_move_set_collision_rejected():
    """Collision in _apply_move_set should return None."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(0, 5),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    # Moving qubit 0 to (0,5) where qubit 1 already sits
    lane = SiteLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    child = tree._apply_move_set(tree.root, move_set)
    assert child is None


def test_collision_filtered_by_generator():
    """ExhaustiveMoveGenerator pre-filters collision-causing rectangles."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(0, 5),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = ExhaustiveMoveGenerator()

    for ms in gen.generate(tree.root, tree):
        for lane in ms:
            if lane.word_id == 0 and lane.site_id == 0 and lane.bus_id == 0:
                src, dst = arch_spec.get_endpoints(lane)
                if tree.root.is_occupied(src):
                    assert not tree.root.is_occupied(dst)


def test_transposition_table_deduplication():
    tree = _make_tree()

    lane_fwd = SiteLaneAddress(0, 0, 0)
    child = tree._apply_move_set(tree.root, frozenset({lane_fwd}))
    assert child is not None

    assert tree.root.config_key in tree.seen
    assert tree.seen[tree.root.config_key].depth == 0


def test_exhaustive_generator_yields_move_sets():
    """ExhaustiveMoveGenerator should yield non-empty frozensets."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator()
    move_sets = list(gen.generate(tree.root, tree))

    assert len(move_sets) > 0
    for ms in move_sets:
        assert isinstance(ms, frozenset)
        assert len(ms) > 0


def test_exhaustive_generator_single_lane_capacity():
    """With capacity 1x1, should yield only single-lane move sets."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)
    move_sets = list(gen.generate(tree.root, tree))

    for ms in move_sets:
        assert len(ms) == 1


def test_exhaustive_generator_no_empty_rectangles():
    """Rectangles with no occupied atoms should not be yielded."""
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 0)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = ExhaustiveMoveGenerator()

    for ms in gen.generate(tree.root, tree):
        encoded_sources = {LocationAddress(lane.word_id, lane.site_id) for lane in ms}
        assert any(tree.root.is_occupied(s) for s in encoded_sources)


def test_expand_node_produces_valid_children():
    """expand_node should produce children with no collisions."""
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator()
    children = tree.expand_node(tree.root, gen)

    assert len(children) > 0
    for child in children:
        assert child.depth == 1
        assert child.parent is tree.root
        # No two qubits at the same location
        locs = list(child.configuration.values())
        assert len(locs) == len(set(locs))


def test_expand_node_deadlock():
    """A stuck configuration should not crash."""
    arch_spec = logical.get_arch_spec()
    placement = {i: LocationAddress(0, i) for i in range(10)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = ExhaustiveMoveGenerator()

    children = tree.expand_node(tree.root, gen)
    assert isinstance(children, list)
