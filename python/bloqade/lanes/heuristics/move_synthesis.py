"""Move synthesis: layout transition to move layers.

Given an architecture spec and two concrete states (before/after layouts),
computes the sequence of move layers. Uses ArchSpec.get_lane_address()
for lane lookup instead of hardcoded bus arithmetic.
"""

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement.lattice import ConcreteState
from bloqade.lanes.layout.path import PathFinder


def _intra_word_moves(
    arch_spec: layout.ArchSpec,
    diffs: list[tuple[layout.LocationAddress, layout.LocationAddress]],
) -> list[tuple[layout.LaneAddress, ...]]:
    """Compute lanes for moves within the same word, grouped by bus_id."""
    bus_moves: dict[int, list[layout.LaneAddress]] = {}
    for before, end in diffs:
        lane = arch_spec.get_lane_address(before, end)
        assert lane is not None, f"No lane from {before} to {end}"
        bus_moves.setdefault(lane.bus_id, []).append(lane)
    return [tuple(lanes) for lanes in bus_moves.values()]


def _compute_move_layers(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
) -> tuple[tuple[layout.LaneAddress, ...], ...]:
    """Compute move layers from state_before to state_after.

    Cross-word moves use PathFinder for multi-hop word bus routing.
    Same-word moves use site bus only.
    """
    diffs = [
        (src, dst)
        for src, dst in zip(state_before.layout, state_after.layout)
        if src != dst
    ]

    # Group by same-word vs cross-word
    same_word: list[tuple[layout.LocationAddress, layout.LocationAddress]] = []
    cross_word: list[tuple[layout.LocationAddress, layout.LocationAddress]] = []
    for src, dst in diffs:
        if src.word_id == dst.word_id:
            same_word.append((src, dst))
        else:
            cross_word.append((src, dst))

    moves: list[tuple[layout.LaneAddress, ...]] = []

    if cross_word:
        path_finder = PathFinder(arch_spec)

        for src, dst in cross_word:
            result = path_finder.find_path(src, dst)
            assert result is not None, f"No path from {src} to {dst}"
            lanes, _ = result
            # Each lane in the path is a separate move layer
            for lane in lanes:
                moves.append((lane,))

    # Same-word moves: site bus only
    for word_id in {s.word_id for s, _ in same_word}:
        word_diffs = [(s, d) for s, d in same_word if s.word_id == word_id]
        moves.extend(_intra_word_moves(arch_spec, word_diffs))

    return tuple(moves)


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
