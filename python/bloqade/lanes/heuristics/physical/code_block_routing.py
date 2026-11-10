"""Soft coordination of participating atoms in inter-block CZ layers.

Generate consistent mover choices for connected, bipartite block interactions.
The existing fixed-target router supplies legal bus schedules. Compare those
schedules with ordinary routing; no block shape is imposed on intermediate
states, and spectators receive no coordination preference.
"""

from __future__ import annotations

from collections import Counter
from itertools import islice, product

from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress


def inter_block_groups(
    state: ConcreteState, pairs: list[tuple[int, int]]
) -> tuple[dict[int, int], tuple[tuple[int, ...], ...]]:
    """Group only participants whose CZ partner belongs to another block."""
    membership = {
        q: block.block_id
        for block in state.code_blocks
        for q in block.members
        if q is not None
    }
    active: dict[int, set[int]] = {}
    for a, b in pairs:
        if a not in membership or b not in membership or membership[a] == membership[b]:
            continue
        for q in (a, b):
            active.setdefault(membership[q], set()).add(q)
    return membership, tuple(
        tuple(sorted(active[block]))
        for block in sorted(active)
        if len(active[block]) > 1
    )


def coordinated_targets(
    arch: ArchSpec,
    state: ConcreteState,
    pairs: list[tuple[int, int]],
    ordinary_goal: dict[int, LocationAddress],
    limit: int,
) -> list[dict[int, LocationAddress]]:
    """Try moving either side of each bipartite block-interaction component.

    Odd-cycle components retain the ordinary goal: there is no assignment of
    a single moving/stationary role to each block in such a component. Intra-
    block and unregistered pairs retain their ordinary targets as well.
    """
    membership, groups = inter_block_groups(state, pairs)
    if not groups:
        return []
    edges: dict[int, set[int]] = {}
    inter_pairs = []
    for a, b in pairs:
        if a in membership and b in membership and membership[a] != membership[b]:
            x, y = membership[a], membership[b]
            edges.setdefault(x, set()).add(y)
            edges.setdefault(y, set()).add(x)
            inter_pairs.append((a, b))
    visited: set[int] = set()
    components: list[dict[int, int]] = []
    for root in sorted(edges):
        if root in visited:
            continue
        colors = {root: 0}
        queue = [root]
        valid = True
        for block in queue:
            visited.add(block)
            for neighbor in sorted(edges[block]):
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[block]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[block]:
                    valid = False
        if valid:
            components.append(colors)
    if not components:
        return []

    candidates: list[dict[int, LocationAddress]] = []
    for flips in islice(product((0, 1), repeat=len(components)), limit):
        moving = {
            block: color == flip
            for component, flip in zip(components, flips)
            for block, color in component.items()
        }
        goal = dict(ordinary_goal)
        valid = True
        for a, b in inter_pairs:
            if membership[a] not in moving:
                continue
            mover, stationary = (a, b) if moving[membership[a]] else (b, a)
            partner = arch.get_cz_partner(state.layout[stationary])
            if partner is None:
                valid = False
                break
            goal[stationary] = state.layout[stationary]
            goal[mover] = partner
        if valid and goal != ordinary_goal and goal not in candidates:
            candidates.append(goal)
    return candidates


def valid_cz_goal(
    arch: ArchSpec,
    state: ConcreteState,
    pairs: list[tuple[int, int]],
    goal: dict[int, LocationAddress],
) -> bool:
    """Require exactly the requested CZ pairs, including occupied spectators."""
    if set(goal) != set(range(len(state.layout))):
        return False
    locations = list(goal.values())
    if arch.check_location_group(locations) or state.occupied.intersection(locations):
        return False
    expected = {frozenset((goal[a], goal[b])) for a, b in pairs}
    atoms = set(locations) | set(state.occupied)
    actual = {
        frozenset((loc, partner))
        for loc in atoms
        if (partner := arch.get_cz_partner(loc)) is not None and partner in atoms
    }
    return actual == expected


def route_score(
    arch: ArchSpec,
    state: ConcreteState,
    groups: tuple[tuple[int, ...], ...],
    result: _native.SolveResult,
) -> tuple[int, int, int] | None:
    """Replay a route; score layers + split-member events, then layers/lanes.

    A split-member event is one participating atom taking a different bus
    action (including staying put) from its block's majority in that shot.
    One event costs one move layer. Return None for an invalid bus schedule.
    """
    layout = list(state.layout)
    splits = 0
    lane_count = 0
    layers = result.move_layers
    for raw_layer in layers:
        lanes = tuple(LaneAddress.from_inner(lane) for lane in raw_layer)
        if arch.check_lane_group(lanes):
            return None
        endpoints = [arch.get_endpoints(lane) for lane in lanes]
        if arch.aod_capacity is not None:
            positions = [arch.get_position(src) for src, _ in endpoints]
            nx, ny = arch.aod_capacity
            if (
                len({p[0] for p in positions}) > nx
                or len({p[1] for p in positions}) > ny
            ):
                return None
        actions = {
            src: (lane.move_type, lane.bus_id, lane.zone_id, lane.direction)
            for lane, (src, _) in zip(lanes, endpoints)
        }
        if state.occupied.intersection(actions):
            return None
        for group in groups:
            counts = Counter(actions.get(layout[q]) for q in group)
            splits += len(group) - max(counts.values())
        destinations = dict(endpoints)
        layout = [destinations.get(loc, loc) for loc in layout]
        if len(set(layout)) != len(layout) or state.occupied.intersection(layout):
            return None
        lane_count += len(lanes)
    goal = {
        qid: LocationAddress.from_inner(loc) for qid, loc in result.goal_config.items()
    }
    if dict(enumerate(layout)) != goal:
        return None
    return len(layers) + splits, len(layers), lane_count
