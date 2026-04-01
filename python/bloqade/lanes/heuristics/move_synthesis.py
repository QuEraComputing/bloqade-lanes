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

    For each qubit that needs to move, finds the path from source to
    destination. Single-hop moves (direct bus connection) are batched
    by bus_id. Multi-hop moves are sequenced individually using PathFinder.
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
        # Try direct single-hop word bus first, fall back to PathFinder
        site_adjustments: list[tuple[layout.LocationAddress, layout.LocationAddress]] = []
        word_bus_lanes: dict[int, list[layout.LaneAddress]] = {}
        multi_hop: list[tuple[layout.LocationAddress, layout.LocationAddress]] = []

        for src, dst in cross_word:
            # Try direct word bus at matching site
            for wb_site in sorted(arch_spec.has_word_buses):
                wb_src = layout.LocationAddress(src.word_id, wb_site)
                wb_dst = layout.LocationAddress(dst.word_id, wb_site)
                lane = arch_spec.get_lane_address(wb_src, wb_dst)
                if lane is not None:
                    # Site adjustment before word bus if needed
                    if src.site_id != wb_site:
                        site_adjustments.append(
                            (src, layout.LocationAddress(src.word_id, wb_site))
                        )
                    word_bus_lanes.setdefault(lane.bus_id, []).append(lane)
                    # Site adjustment after word bus if needed
                    if wb_site != dst.site_id:
                        same_word.append(
                            (layout.LocationAddress(dst.word_id, wb_site), dst)
                        )
                    break
            else:
                # No direct word bus — need multi-hop
                multi_hop.append((src, dst))

        # Execute site adjustments first
        if site_adjustments:
            for word_id in {s.word_id for s, _ in site_adjustments}:
                word_diffs = [
                    (s, d) for s, d in site_adjustments if s.word_id == word_id
                ]
                moves.extend(_intra_word_moves(arch_spec, word_diffs))

        # Then single-hop word bus moves (batched by bus)
        for lanes in word_bus_lanes.values():
            moves.append(tuple(lanes))

        # Then multi-hop moves (sequenced individually via PathFinder)
        if multi_hop:
            # Collect all occupied positions to avoid collisions
            all_positions = set(state_before.layout) | set(state_after.layout)
            path_finder = PathFinder(arch_spec)
            for src, dst in multi_hop:
                occupied = frozenset(all_positions - {src, dst})
                result = path_finder.find_path(src, dst, occupied=occupied)
                assert result is not None, f"No path from {src} to {dst}"
                lanes, _ = result
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
