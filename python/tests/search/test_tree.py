"""Tests for ConfigurationTree and ExhaustiveMoveGenerator."""

import pytest

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
)
from bloqade.lanes.search.generators import ExhaustiveMoveGenerator, MoveGenerator
from bloqade.lanes.search.tree import (
    ConfigurationTree,
    ExpansionStatus,
    InvalidMoveError,
)


def _make_tree() -> ConfigurationTree:
    """Create a tree with the logical Gemini arch spec."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(4, 0),
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

    lane = WordLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    child = tree.apply_move_set(tree.root, move_set, strict=False)
    assert child is not None
    assert child.depth == 1
    assert child.parent is tree.root
    assert child.parent_moves == move_set
    assert child.configuration[0] == LocationAddress(1, 0)
    assert child.configuration[1] == LocationAddress(4, 0)


def test_apply_move_set_collision_strict_raises():
    """In strict mode, collisions raise InvalidMoveError."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    lane = WordLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    with pytest.raises(InvalidMoveError, match="Collision"):
        tree.apply_move_set(tree.root, move_set, strict=True)


def test_apply_move_set_collision_nonstrict_returns_none():
    """In non-strict mode, collisions return None."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    lane = WordLaneAddress(0, 0, 0)
    move_set = frozenset({lane})

    child = tree.apply_move_set(tree.root, move_set, strict=False)
    assert child is None


def test_collision_filtered_by_generator():
    """ExhaustiveMoveGenerator pre-filters collision-causing grids."""
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
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

    lane_fwd = WordLaneAddress(0, 0, 0)
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


def test_exhaustive_generator_no_empty_grids():
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 0)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = ExhaustiveMoveGenerator()

    for ms in gen.generate(tree.root, tree):
        encoded_sources = {LocationAddress(lane.word_id, lane.site_id, lane.zone_id) for lane in ms}
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
    # Fill all 20 words × 1 site = 20 locations to create a deadlock
    placement = {i: LocationAddress(i, 0) for i in range(20)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = ExhaustiveMoveGenerator()

    children = tree.expand_node(tree.root, gen, strict=False)
    assert isinstance(children, list)


class _FixedMoveGenerator:
    def __init__(self, move_sets: list[frozenset[LaneAddress]]):
        self._move_sets = move_sets

    def generate(self, _node, _tree):  # type: ignore[no-untyped-def]
        yield from self._move_sets


def test_try_move_set_reports_created_child():
    tree = _make_tree()
    move_set = frozenset({WordLaneAddress(0, 0, 0)})
    outcome = tree.try_move_set(tree.root, move_set, strict=False)
    assert outcome.status == ExpansionStatus.CREATED_CHILD
    assert outcome.child is not None
    assert outcome.child.parent is tree.root
    assert outcome.existing_node is None


def test_try_move_set_reports_already_child():
    tree = _make_tree()
    move_set = frozenset({WordLaneAddress(0, 0, 0)})
    first = tree.try_move_set(tree.root, move_set, strict=False)
    assert first.status == ExpansionStatus.CREATED_CHILD
    second = tree.try_move_set(tree.root, move_set, strict=False)
    assert second.status == ExpansionStatus.ALREADY_CHILD
    assert second.child is first.child


def test_try_move_set_reports_transposition_seen():
    tree = _make_tree()
    outcome = tree.try_move_set(tree.root, frozenset(), strict=False)
    assert outcome.status == ExpansionStatus.TRANSPOSITION_SEEN
    assert outcome.child is None
    assert outcome.existing_node is tree.root


def test_try_move_set_reports_collision():
    arch_spec = logical.get_arch_spec()
    placement = {
        0: LocationAddress(0, 0),
        1: LocationAddress(1, 0),
    }
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    move_set = frozenset({WordLaneAddress(0, 0, 0)})
    outcome = tree.try_move_set(tree.root, move_set, strict=False)
    assert outcome.status == ExpansionStatus.COLLISION
    assert outcome.child is None


def test_try_move_set_reports_collision_with_blocked_locations():
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 0)}
    blocked = frozenset({LocationAddress(1, 0)})
    tree = ConfigurationTree(
        arch_spec=arch_spec,
        root=ConfigurationTree.from_initial_placement(arch_spec, placement).root,
        blocked_locations=blocked,
    )
    move_set = frozenset({WordLaneAddress(0, 0, 0)})
    outcome = tree.try_move_set(tree.root, move_set, strict=False)
    assert outcome.status == ExpansionStatus.COLLISION
    assert outcome.child is None


def test_try_move_set_reports_invalid_lane():
    tree = _make_tree()
    invalid = LaneAddress(MoveType.SITE, 999, 999, 999, Direction.FORWARD)
    outcome = tree.try_move_set(tree.root, frozenset({invalid}), strict=False)
    assert outcome.status == ExpansionStatus.INVALID_LANE
    assert outcome.child is None


def test_expand_node_detailed_reports_all_attempts():
    tree = _make_tree()
    move_set = frozenset({WordLaneAddress(0, 0, 0)})
    invalid = frozenset({LaneAddress(MoveType.SITE, 999, 999, 999, Direction.FORWARD)})
    gen: MoveGenerator = _FixedMoveGenerator([move_set, move_set, invalid])  # type: ignore[assignment]
    outcomes = list(tree.expand_node_detailed(tree.root, gen, strict=False))
    statuses = [outcome.status for outcome in outcomes]
    assert statuses == [
        ExpansionStatus.CREATED_CHILD,
        ExpansionStatus.ALREADY_CHILD,
        ExpansionStatus.INVALID_LANE,
    ]


def test_expand_node_compat_filters_to_created_children_only():
    tree = _make_tree()
    move_set = frozenset({WordLaneAddress(0, 0, 0)})
    invalid = frozenset({LaneAddress(MoveType.SITE, 999, 999, 999, Direction.FORWARD)})
    gen: MoveGenerator = _FixedMoveGenerator([move_set, move_set, invalid])  # type: ignore[assignment]
    children = tree.expand_node(tree.root, gen, strict=False)
    assert len(children) == 1


def test_valid_lanes_returns_nonempty():
    tree = _make_tree()
    lanes = frozenset(tree.valid_lanes(tree.root))
    assert len(lanes) > 0
    # All lanes should have occupied src and unoccupied dst
    for lane in lanes:
        src, dst = tree.arch_spec.get_endpoints(lane)
        assert tree.root.is_occupied(src)
        assert not tree.root.is_occupied(dst)


def test_valid_lanes_excludes_blocked_destinations():
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 0)}
    blocked = frozenset({LocationAddress(1, 0)})
    tree = ConfigurationTree(
        arch_spec=arch_spec,
        root=ConfigurationTree.from_initial_placement(arch_spec, placement).root,
        blocked_locations=blocked,
    )
    lanes = frozenset(tree.valid_lanes(tree.root))
    assert len(lanes) > 0
    for lane in lanes:
        _, dst = tree.arch_spec.get_endpoints(lane)
        assert dst not in blocked


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


def test_apply_move_set_rejects_inconsistent_group():
    """A lane group with mixed move types raises InvalidMoveError."""
    tree = _make_tree()

    # Mixing site bus and word bus lanes violates lane-group consistency.
    inconsistent_group = frozenset(
        {
            SiteLaneAddress(0, 0, 0),
            WordLaneAddress(0, 0, 0),
        }
    )

    with pytest.raises(InvalidMoveError, match="lane-group validation"):
        tree.apply_move_set(tree.root, inconsistent_group)


def test_valid_lanes_no_collisions():
    """Destinations of valid lanes must be unoccupied."""
    arch_spec = logical.get_arch_spec()
    # Place atoms at word 0 and word 1 so forward bus-0 from word 0 collides
    placement = {0: LocationAddress(0, 0), 1: LocationAddress(1, 0)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    from bloqade.lanes.layout import MoveType

    word_lanes = list(tree.valid_lanes(tree.root, move_type=MoveType.WORD))
    # All valid lanes must have unoccupied destinations
    for lane in word_lanes:
        src, dst = tree.arch_spec.get_endpoints(lane)
        assert not tree.root.is_occupied(dst)
