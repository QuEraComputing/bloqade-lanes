"""Move synthesis: layout transition to move layers.

Given an architecture spec and two concrete states (before/after layouts),
computes the sequence of move layers. Uses ArchSpec.get_lane_address()
for lane lookup instead of hardcoded bus arithmetic.
"""

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement.lattice import ConcreteState
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.search import ConfigurationNode, ConfigurationTree

from .physical_placement import EntropyPlacementTraversal


def _intra_word_moves(
    arch_spec: layout.ArchSpec,
    diffs: list[tuple[layout.LocationAddress, layout.LocationAddress]],
    path_finder: PathFinder | None = None,
) -> list[tuple[layout.LaneAddress, ...]]:
    """Compute lanes for moves within the same word, grouped by bus_id.

    Falls back to PathFinder for multi-hop site bus moves (e.g.,
    HypercubeSiteTopology where not all site pairs are directly connected).
    """
    bus_moves: dict[int, list[layout.LaneAddress]] = {}
    multi_hop_lanes: list[tuple[layout.LaneAddress, ...]] = []
    for before, end in diffs:
        lane = arch_spec.get_lane_address(before, end)
        if lane is not None:
            bus_moves.setdefault(lane.bus_id, []).append(lane)
        else:
            # Multi-hop intra-word move
            if path_finder is None:
                path_finder = PathFinder(arch_spec)
            result = path_finder.find_path(before, end)
            assert result is not None, f"No path from {before} to {end}"
            lanes, _ = result
            for hop in lanes:
                multi_hop_lanes.append((hop,))
    return [tuple(lanes) for lanes in bus_moves.values()] + multi_hop_lanes


def _compute_move_layers(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
) -> tuple[tuple[layout.LaneAddress, ...], ...]:
    """Compute move layers from state_before to state_after.

    For each qubit that needs to move, finds the path from source to
    destination. Single-hop moves (direct bus connection) are batched
    by bus_id. Multi-hop moves are sequenced individually using PathFinder.
    """
    target = {
        atom_id: dst
        for (atom_id, src), dst in zip(
            enumerate(state_before.layout), state_after.layout
        )
        if src != dst
    }
    root_node = ConfigurationNode(
        dict(enumerate(state_before.layout)), external_occupied=state_before.occupied
    )
    tree = ConfigurationTree(arch_spec, root_node, state_before.occupied)

    traversal = EntropyPlacementTraversal()
    result = traversal.path_to_target_config(tree=tree, target=target)

    assert result.goal_node, "no solution found"
    return result.goal_nodes[0].to_move_program()


# Public API
compute_move_layers = _compute_move_layers


def move_to_entangle(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...]]:
    """Synthesize move layers from current layout to CZ entangling layout."""
    return state_after, compute_move_layers(arch_spec, state_before, state_after)


def move_to_left(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...]]:
    """Synthesize move layers from CZ layout to post-CZ return layout."""
    forward_layers = compute_move_layers(arch_spec, state_after, state_before)
    reverse_layers = tuple(
        tuple(lane.reverse() for lane in move_lanes[::-1])
        for move_lanes in forward_layers[::-1]
    )
    return state_after, reverse_layers
