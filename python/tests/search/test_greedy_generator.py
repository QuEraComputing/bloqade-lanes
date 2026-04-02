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
        placement = {0: LocationAddress(0, 0), 1: LocationAddress(1, 0)}
    if target is None:
        target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    tree = ConfigurationTree.from_initial_placement(
        arch_spec, placement, blocked_locations=blocked
    )
    gen = GreedyMoveGenerator(target=target)
    return gen, tree


# --- Protocol compliance ---


def test_satisfies_protocol():
    gen, _ = _make_setup()
    assert isinstance(gen, MoveGenerator)


# --- _compute_path_lengths ---


def test_compute_path_lengths_finds_unresolved():
    """Qubits not at target are identified with their path lengths."""
    gen, tree = _make_setup(
        placement={0: LocationAddress(0, 0)},
        target={0: LocationAddress(0, 5)},
    )
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    assert 0 in unresolved
    assert unresolved[0] == LocationAddress(0, 0)
    assert path_lengths[0] > 0


def test_compute_path_lengths_skips_resolved():
    """Qubits already at target are not included."""
    gen, tree = _make_setup(
        placement={0: LocationAddress(0, 5)},
        target={0: LocationAddress(0, 5)},
    )
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    assert 0 not in unresolved
    assert 0 not in path_lengths


def test_compute_path_lengths_skips_qubits_not_in_target():
    """Qubits not in the target dict are not included."""
    gen, tree = _make_setup(
        placement={0: LocationAddress(0, 0), 1: LocationAddress(1, 0)},
        target={0: LocationAddress(0, 5)},
    )
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    assert 0 in unresolved
    assert 1 not in unresolved


# --- _compute_next_hops ---


def test_compute_next_hops_returns_lanes():
    """Each unresolved qubit gets a next-hop lane."""
    gen, tree = _make_setup(
        placement={0: LocationAddress(0, 0)},
        target={0: LocationAddress(0, 5)},
    )
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    assert 0 in next_hops
    assert isinstance(next_hops[0], LaneAddress)


def test_compute_next_hops_longest_path_first():
    """Qubits with longer paths are processed first."""
    placement = {0: LocationAddress(0, 0), 1: LocationAddress(1, 0)}
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    gen, tree = _make_setup(placement=placement, target=target)
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)

    assert 0 in path_lengths
    assert 1 in path_lengths

    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    assert 0 in next_hops
    assert 1 in next_hops


def test_compute_next_hops_updates_occupied():
    """Later atoms route around earlier atoms' committed moves."""
    placement = {0: LocationAddress(0, 0), 1: LocationAddress(0, 1)}
    target = {0: LocationAddress(0, 5), 1: LocationAddress(0, 3)}
    gen, tree = _make_setup(placement=placement, target=target)
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)

    for qid in unresolved:
        if qid in next_hops:
            assert isinstance(next_hops[qid], LaneAddress)


# --- _group_by_triplet ---


def test_group_by_triplet_groups_same_bus():
    """Lanes with the same (move_type, bus_id, direction) are grouped."""
    lane_a = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD)
    lane_b = LaneAddress(MoveType.SITE, 1, 0, 0, Direction.FORWARD)
    lane_c = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD)

    next_hops = {0: lane_a, 1: lane_b, 2: lane_c}
    groups = GreedyMoveGenerator._group_by_triplet(next_hops)

    site_key = (MoveType.SITE, 0, Direction.FORWARD)
    word_key = (MoveType.WORD, 0, Direction.FORWARD)

    assert site_key in groups
    assert word_key in groups
    assert len(groups[site_key]) == 2
    assert len(groups[word_key]) == 1


def test_group_by_triplet_separates_directions():
    """Lanes with different directions are in separate groups."""
    lane_fwd = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD)
    lane_bwd = LaneAddress(MoveType.SITE, 1, 0, 0, Direction.BACKWARD)

    groups = GreedyMoveGenerator._group_by_triplet({0: lane_fwd, 1: lane_bwd})
    assert len(groups) == 2


def test_group_by_triplet_empty():
    """Empty input produces empty groups."""
    groups = GreedyMoveGenerator._group_by_triplet({})
    assert groups == {}


# --- _is_valid_rect ---


def test_is_valid_rect_all_present():
    """Returns True when all positions map to non-collision sources."""
    ctx = _make_bus_context(
        {
            (0.0, 0.0): LocationAddress(0, 0),
            (0.0, 1.0): LocationAddress(0, 1),
            (1.0, 0.0): LocationAddress(1, 0),
            (1.0, 1.0): LocationAddress(1, 1),
        }
    )
    assert ctx.is_valid_rect({0.0, 1.0}, {0.0, 1.0}) is True


def test_is_valid_rect_missing_position():
    """Returns False when a position has no bus source."""
    ctx = _make_bus_context(
        {
            (0.0, 0.0): LocationAddress(0, 0),
            # (0.0, 1.0) missing
            (1.0, 0.0): LocationAddress(1, 0),
            (1.0, 1.0): LocationAddress(1, 1),
        }
    )
    assert ctx.is_valid_rect({0.0, 1.0}, {0.0, 1.0}) is False


def test_is_valid_rect_collision():
    """Returns False when a position is a collision source."""
    loc = LocationAddress(0, 0)
    ctx = _make_bus_context({(0.0, 0.0): loc}, frozenset({loc}))
    assert ctx.is_valid_rect({0.0}, {0.0}) is False


# --- BusContext.rect_to_lanes ---


def test_rect_to_lanes_builds_complete_grid():
    """Produces one lane per position in the rectangle."""
    ctx = _make_bus_context(
        {
            (0.0, 0.0): LocationAddress(0, 0),
            (0.0, 1.0): LocationAddress(0, 1),
            (1.0, 0.0): LocationAddress(1, 0),
            (1.0, 1.0): LocationAddress(1, 1),
        }
    )
    lanes = ctx.rect_to_lanes({0.0, 1.0}, {0.0, 1.0})
    assert len(lanes) == 4
    assert all(isinstance(lane, LaneAddress) for lane in lanes)


def test_rect_to_lanes_skips_missing_positions():
    """Positions not in pos_to_loc are silently skipped."""
    ctx = _make_bus_context({(0.0, 0.0): LocationAddress(0, 0)})
    lanes = ctx.rect_to_lanes({0.0, 1.0}, {0.0})
    assert len(lanes) == 1


# --- _greedy_init ---


def test_greedy_init_forms_clusters():
    """Greedy init should produce at least one cluster from valid entries."""
    gen, tree = _make_setup()
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    groups = gen._group_by_triplet(next_hops)

    for (mt, bid, d), entries in groups.items():
        ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
        clusters = ctx.greedy_init(entries)
        assert len(clusters) >= 1
        for xs, ys in clusters:
            assert len(xs) >= 1
            assert len(ys) >= 1


def test_greedy_init_all_valid_clusters():
    """Each cluster from greedy init should be a valid rectangle."""
    gen, tree = _make_setup()
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    groups = gen._group_by_triplet(next_hops)

    for (mt, bid, d), entries in groups.items():
        ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
        clusters = ctx.greedy_init(entries)
        for xs, ys in clusters:
            assert ctx.is_valid_rect(xs, ys)


# --- _merge_clusters ---


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


def test_merge_clusters_merges_compatible():
    """Two compatible 1x1 clusters should be merged into one."""
    loc_a = LocationAddress(0, 0)
    loc_b = LocationAddress(0, 1)
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
    loc_a = LocationAddress(0, 0)
    loc_b = LocationAddress(1, 1)
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
    loc_a = LocationAddress(0, 0)
    loc_b = LocationAddress(0, 1)
    loc_c = LocationAddress(1, 0)
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


# --- _build_aod_grids (divide and conquer) ---


def test_build_aod_grids_single_atom():
    """A single atom produces a valid 1x1 grid."""
    arch_spec = logical.get_arch_spec()
    placement = {0: LocationAddress(0, 0)}
    target = {0: LocationAddress(0, 5)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    gen = GreedyMoveGenerator(target=target)

    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    groups = gen._group_by_triplet(next_hops)

    for (mt, bid, d), entries in groups.items():
        ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
        grids = ctx.build_aod_grids(entries)
        assert len(grids) >= 1
        for grid in grids:
            assert len(grid) >= 1


def test_build_aod_grids_validates_rectangle():
    """Each produced grid should pass arch_spec lane group validation."""
    gen, tree = _make_setup()
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    groups = gen._group_by_triplet(next_hops)

    for (mt, bid, d), entries in groups.items():
        ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
        for grid in ctx.build_aod_grids(entries):
            errors = tree.arch_spec.check_lane_group(list(grid))
            assert len(errors) == 0, f"Grid failed validation: {errors}"


def test_build_aod_grids_no_destination_collisions():
    """No occupied source in any grid should have an occupied destination."""
    gen, tree = _make_setup()
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    groups = gen._group_by_triplet(next_hops)

    for (mt, bid, d), entries in groups.items():
        ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
        for grid in ctx.build_aod_grids(entries):
            for lane in grid:
                src, dst = tree.arch_spec.get_endpoints(lane)
                if src in occupied:
                    assert (
                        dst not in occupied
                    ), f"Collision: occupied src {src} -> occupied dst {dst}"


def test_build_aod_grids_merges_compatible_clusters():
    """Two atoms that can form a valid rectangle should be merged."""
    gen, tree = _make_setup()
    occupied = tree.root.occupied_locations | tree.blocked_locations
    unresolved, path_lengths = gen._compute_path_lengths(tree.root, occupied, tree)
    next_hops = gen._compute_next_hops(unresolved, path_lengths, occupied, tree)
    groups = gen._group_by_triplet(next_hops)

    # At least one group should have merged clusters (grid with >1 lane)
    found_merged = False
    for (mt, bid, d), entries in groups.items():
        if len(entries) < 2:
            continue
        ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
        for grid in ctx.build_aod_grids(entries):
            if len(grid) > 1:
                found_merged = True
    assert found_merged, "Expected at least one merged cluster with multiple lanes"


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
    placement = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    target = {0: LocationAddress(0, 5), 1: LocationAddress(1, 5)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))
    assert moves == []


def test_generate_moves_advance_toward_target():
    """The first yielded move set should move at least one qubit closer."""
    placement = {0: LocationAddress(0, 0)}
    target = {0: LocationAddress(0, 5)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) >= 1

    first_moveset = moves[0]
    child = tree.apply_move_set(tree.root, first_moveset, strict=False)
    assert child is not None
    assert child.configuration[0] != LocationAddress(0, 0)


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
    placement = {0: LocationAddress(0, 0), 1: LocationAddress(1, 0)}
    target = {0: LocationAddress(0, 5)}
    gen, tree = _make_setup(placement=placement, target=target)
    moves = list(gen.generate(tree.root, tree))
    assert len(moves) >= 1

    for ms in moves:
        for lane in ms:
            src, _ = tree.arch_spec.get_endpoints(lane)
            assert src == LocationAddress(0, 0)
