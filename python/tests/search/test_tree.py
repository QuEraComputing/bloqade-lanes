"""Tests for ConfigurationTree and ExhaustiveMoveGenerator."""

import pytest

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress, SiteLaneAddress
from bloqade.lanes.search.generators import ExhaustiveMoveGenerator
from bloqade.lanes.search.tree import ConfigurationTree, InvalidMoveError


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


def test_apply_move_set_valid():
    tree = _make_tree()

    lane = SiteLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    child = tree.apply_move_set(tree.root, move_set, strict=False)
    assert child is not None
    assert child.depth == 1
    assert child.parent is tree.root
    assert child.parent_moves == move_set
    assert child.configuration[0] == LocationAddress(0, 5)
    assert child.configuration[1] == LocationAddress(1, 0)


def test_apply_move_set_collision_strict_raises():
    """In strict mode, collisions raise InvalidMoveError."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(0, 5),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    lane = SiteLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    with pytest.raises(InvalidMoveError, match="Collision"):
        tree.apply_move_set(tree.root, move_set, strict=True)


def test_apply_move_set_collision_nonstrict_returns_none():
    """In non-strict mode, collisions return None."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(0, 5),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    lane = SiteLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    child = tree.apply_move_set(tree.root, move_set, strict=False)
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
    child = tree.apply_move_set(tree.root, frozenset({lane_fwd}), strict=False)
    assert child is not None

    assert tree.root.config_key in tree.seen
    assert tree.seen[tree.root.config_key].depth == 0


def test_exhaustive_generator_yields_move_sets():
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator()
    move_sets = list(gen.generate(tree.root, tree))

    assert len(move_sets) > 0
    for ms in move_sets:
        assert isinstance(ms, frozenset)
        assert len(ms) > 0


def test_exhaustive_generator_single_lane_capacity():
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator(max_x_capacity=1, max_y_capacity=1)
    move_sets = list(gen.generate(tree.root, tree))

    for ms in move_sets:
        assert len(ms) == 1


def test_exhaustive_generator_no_empty_rectangles():
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 0)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = ExhaustiveMoveGenerator()

    for ms in gen.generate(tree.root, tree):
        encoded_sources = {LocationAddress(lane.word_id, lane.site_id) for lane in ms}
        assert any(tree.root.is_occupied(s) for s in encoded_sources)


def test_expand_produces_valid_children():
    tree = _make_tree()
    gen = ExhaustiveMoveGenerator()
    children = tree.expand_node(tree.root, gen, strict=False)

    assert len(children) > 0
    for child in children:
        assert child.depth == 1
        assert child.parent is tree.root
        locs = list(child.configuration.values())
        assert len(locs) == len(set(locs))


def test_expand_deadlock():
    arch_spec = logical.get_arch_spec()
    placement = {i: LocationAddress(0, i) for i in range(10)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = ExhaustiveMoveGenerator()

    children = tree.expand_node(tree.root, gen, strict=False)
    assert isinstance(children, list)


def test_valid_lanes_returns_nonempty():
    tree = _make_tree()
    lanes = frozenset(tree.valid_lanes(tree.root))
    assert len(lanes) > 0
    # All lanes should have occupied src and unoccupied dst
    for lane in lanes:
        src, dst = tree.arch_spec.get_endpoints(lane)
        assert tree.root.is_occupied(src)
        assert not tree.root.is_occupied(dst)


def test_valid_lanes_filter_by_move_type():
    from bloqade.lanes.layout import MoveType

    tree = _make_tree()
    site_lanes = frozenset(tree.valid_lanes(tree.root, move_type=MoveType.SITE))
    word_lanes = frozenset(tree.valid_lanes(tree.root, move_type=MoveType.WORD))
    all_lanes = frozenset(tree.valid_lanes(tree.root))

    # Filtered sets should be subsets of all
    assert site_lanes <= all_lanes
    assert word_lanes <= all_lanes
    # All lanes should be of the correct type
    for lane in site_lanes:
        assert lane.move_type == MoveType.SITE
    for lane in word_lanes:
        assert lane.move_type == MoveType.WORD


def test_valid_lanes_filter_by_direction():
    from bloqade.lanes.layout import Direction

    tree = _make_tree()
    fwd = list(tree.valid_lanes(tree.root, direction=Direction.FORWARD))
    bwd = list(tree.valid_lanes(tree.root, direction=Direction.BACKWARD))

    for lane in fwd:
        assert lane.direction == Direction.FORWARD
    for lane in bwd:
        assert lane.direction == Direction.BACKWARD


def test_valid_lanes_no_collisions():
    """All sites occupied — no valid lanes with unoccupied destinations."""
    arch_spec = logical.get_arch_spec()
    placement = {i: LocationAddress(0, i) for i in range(10)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    # Site bus lanes within word 0 should all be blocked
    from bloqade.lanes.layout import MoveType

    site_lanes = tree.valid_lanes(tree.root, move_type=MoveType.SITE)
    # May still have word bus lanes, but site bus on word 0 should be empty
    for lane in site_lanes:
        src, dst = tree.arch_spec.get_endpoints(lane)
        assert not tree.root.is_occupied(dst)
