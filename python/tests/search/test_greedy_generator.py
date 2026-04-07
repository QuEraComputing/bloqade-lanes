"""Tests for GreedyMoveGenerator."""

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import Direction, LaneAddress, LocationAddress, MoveType
from bloqade.lanes.search.generators import (
    BusContext,
    GreedyMoveGenerator,
    MoveGenerator,
)
from bloqade.lanes.search.tree import ConfigurationTree


def _make_setup(
    placement: dict[int, LocationAddress] | None = None,
    target: dict[int, LocationAddress] | None = None,
    blocked: frozenset[LocationAddress] = frozenset(),
):
    arch_spec = logical.get_arch_spec()
    if placement is None:
        placement = {0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 1, 0)}
    if target is None:
        target = {0: LocationAddress(0, 4, 0), 1: LocationAddress(0, 5, 0)}
    tree = ConfigurationTree.from_initial_placement(
        arch_spec, placement, blocked_locations=blocked
    )
    gen = GreedyMoveGenerator(target=target)
    return gen, tree


# --- Protocol compliance ---


def test_satisfies_protocol():
    gen, _ = _make_setup()
    assert isinstance(gen, MoveGenerator)


# --- BusContext primitives ---


def _make_bus_context(
    pos_to_loc: dict[tuple[float, float], LocationAddress],
    collision_srcs: frozenset[LocationAddress] = frozenset(),
) -> BusContext:
    """Create a minimal BusContext for unit tests."""
    arch_spec = logical.get_arch_spec()
    return BusContext(
        move_type=MoveType.SITE,
        bus_id=0,
        direction=Direction.FORWARD,
        arch_spec=arch_spec,
        pos_to_loc=pos_to_loc,
        collision_srcs=collision_srcs,
    )


def test_is_valid_rect_all_present():
    """Returns True when all positions map to non-collision sources."""
    ctx = _make_bus_context(
        {
            (0.0, 0.0): LocationAddress(0, 0, 0),
            (0.0, 1.0): LocationAddress(0, 0, 1),
            (1.0, 0.0): LocationAddress(0, 1, 0),
            (1.0, 1.0): LocationAddress(0, 1, 1),
        }
    )
    assert ctx.is_valid_rect({0.0, 1.0}, {0.0, 1.0}) is True


def test_is_valid_rect_missing_position():
    """Returns False when a position has no bus source."""
    ctx = _make_bus_context(
        {
            (0.0, 0.0): LocationAddress(0, 0, 0),
            # (0.0, 1.0) missing
            (1.0, 0.0): LocationAddress(0, 1, 0),
            (1.0, 1.0): LocationAddress(0, 1, 1),
        }
    )
    assert ctx.is_valid_rect({0.0, 1.0}, {0.0, 1.0}) is False


def test_is_valid_rect_collision():
    """Returns False when a position is a collision source."""
    loc = LocationAddress(0, 0, 0)
    ctx = _make_bus_context({(0.0, 0.0): loc}, frozenset({loc}))
    assert ctx.is_valid_rect({0.0}, {0.0}) is False


# --- BusContext.rect_to_lanes ---


def test_rect_to_lanes_builds_complete_grid():
    """Produces one lane per position in the rectangle."""
    ctx = _make_bus_context(
        {
            (0.0, 0.0): LocationAddress(0, 0, 0),
            (0.0, 1.0): LocationAddress(0, 0, 1),
            (1.0, 0.0): LocationAddress(0, 1, 0),
            (1.0, 1.0): LocationAddress(0, 1, 1),
        }
    )
    lanes = ctx.rect_to_lanes({0.0, 1.0}, {0.0, 1.0})
    assert len(lanes) == 4
    assert all(isinstance(lane, LaneAddress) for lane in lanes)


def test_rect_to_lanes_skips_missing_positions():
    """Positions not in pos_to_loc are silently skipped."""
    ctx = _make_bus_context({(0.0, 0.0): LocationAddress(0, 0, 0)})
    lanes = ctx.rect_to_lanes({0.0, 1.0}, {0.0})
    assert len(lanes) == 1


# --- merge_clusters ---


def test_merge_clusters_merges_compatible():
    """Two compatible 1x1 clusters should be merged into one."""
    loc_a = LocationAddress(0, 0, 0)
    loc_b = LocationAddress(0, 0, 1)
    ctx = _make_bus_context({(0.0, 0.0): loc_a, (0.0, 1.0): loc_b})
    clusters: list[tuple[set[float], set[float]]] = [
        ({0.0}, {0.0}),
        ({0.0}, {1.0}),
    ]
    solved = ctx.merge_clusters(clusters)
    assert len(solved) == 1
    xs, ys = solved[0]
    assert xs == {0.0}
    assert ys == {0.0, 1.0}


def test_merge_clusters_incompatible_stay_separate():
    """Clusters that can't merge remain as separate solved clusters."""
    loc_a = LocationAddress(0, 0, 0)
    loc_b = LocationAddress(0, 1, 1)
    # Missing (0.0, 1.0) and (1.0, 0.0) means the 2x2 rect is invalid
    ctx = _make_bus_context({(0.0, 0.0): loc_a, (1.0, 1.0): loc_b})
    clusters: list[tuple[set[float], set[float]]] = [
        ({0.0}, {0.0}),
        ({1.0}, {1.0}),
    ]
    solved = ctx.merge_clusters(clusters)
    assert len(solved) == 2


def test_merge_clusters_solves_non_participants():
    """Clusters that don't participate in any merge are promoted to solved."""
    loc_a = LocationAddress(0, 0, 0)
    loc_b = LocationAddress(0, 0, 1)
    loc_c = LocationAddress(0, 1, 0)
    # a and b can merge (same x), but c can't join (missing (1,1))
    ctx = _make_bus_context(
        {
            (0.0, 0.0): loc_a,
            (0.0, 1.0): loc_b,
            (2.0, 0.0): loc_c,
        }
    )
    clusters: list[tuple[set[float], set[float]]] = [
        ({0.0}, {0.0}),
        ({0.0}, {1.0}),
        ({2.0}, {0.0}),
    ]
    solved = ctx.merge_clusters(clusters)
    # a+b merged, c stays separate
    assert len(solved) == 2


# --- Full generate integration ---


def test_generate_yields_frozensets():
    gen, tree = _make_setup()
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) > 0
    for ms in moves:
        assert isinstance(ms, frozenset)
        assert all(isinstance(lane, LaneAddress) for lane in ms)
        assert len(ms) > 0


def test_generate_all_resolved_yields_nothing():
    """When all qubits are already at target, no candidates are generated."""
    placement = {0: LocationAddress(0, 4, 0), 1: LocationAddress(0, 5, 0)}
    target = {0: LocationAddress(0, 4, 0), 1: LocationAddress(0, 5, 0)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))
    assert moves == []


def test_generate_moves_advance_toward_target():
    """The first yielded move set should move at least one qubit closer."""
    placement = {0: LocationAddress(0, 0, 0)}
    target = {0: LocationAddress(0, 4, 0)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) >= 1

    first_moveset = moves[0]
    child = tree.apply_move_set(tree.root, first_moveset, strict=False)
    assert child is not None
    assert child.configuration[0] != LocationAddress(0, 0, 0)


def test_generate_expand_node_integration():
    """GreedyMoveGenerator should work with ConfigurationTree.expand_node."""
    gen, tree = _make_setup()
    children = tree.expand_node(tree.root, gen, strict=False)
    assert len(children) > 0
    for child in children:
        assert child.depth == 1
        assert child.parent is tree.root


def test_generate_qubit_not_in_target_is_ignored():
    """Qubits not in the target dict should not generate moves."""
    placement = {0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 1, 0)}
    target = {0: LocationAddress(0, 4, 0)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) >= 1

    for ms in moves:
        for lane in ms:
            src, _ = tree.arch_spec.get_endpoints(lane)
            assert src == LocationAddress(0, 0, 0)


def test_generate_single_atom():
    """A single atom produces at least one valid moveset."""
    placement = {0: LocationAddress(0, 0, 0)}
    target = {0: LocationAddress(0, 4, 0)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) >= 1
    for grid in moves:
        assert len(grid) >= 1


def test_generate_grids_validate():
    """Each produced grid should pass arch_spec lane group validation."""
    gen, tree = _make_setup()
    for grid in gen.generate(tree.root, tree):
        errors = tree.arch_spec.check_lane_group(list(grid))
        assert len(errors) == 0, f"Grid failed validation: {errors}"


def test_generate_no_destination_collisions():
    """No occupied source in any grid should have an occupied destination."""
    gen, tree = _make_setup()
    occupied = tree.root.occupied_locations | tree.blocked_locations
    for grid in gen.generate(tree.root, tree):
        for lane in grid:
            src, dst = tree.arch_spec.get_endpoints(lane)
            if src in occupied:
                assert (
                    dst not in occupied
                ), f"Collision: occupied src {src} -> occupied dst {dst}"


def test_generate_multiple_atoms_can_merge():
    """Two atoms in the same word that can share a site bus move should merge."""
    # Place both atoms in the same word — site bus moves operate within a word
    # and can merge atoms that share the same bus pattern.
    placement = {0: LocationAddress(0, 0, 0), 1: LocationAddress(0, 0, 1)}
    target = {0: LocationAddress(0, 2, 0), 1: LocationAddress(0, 2, 1)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))

    found_merged = any(len(grid) > 1 for grid in moves)
    assert found_merged, "Expected at least one grid with multiple lanes"
