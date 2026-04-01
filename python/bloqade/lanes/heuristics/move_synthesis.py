"""Move synthesis: layout transition to move layers.

Given an architecture spec and two concrete states (before/after layouts),
computes the sequence of move layers. Uses ArchSpec.get_lane_address()
for lane lookup instead of hardcoded bus arithmetic.
"""

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement.lattice import ConcreteState


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

    Cross-word moves: site bus adjustment first (within starting word),
    then word bus transfer. Same-word moves: site bus only.
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

    # Cross-word moves: site bus first (rearrange within starting word),
    # then word bus (transfer to destination word)
    if cross_word:
        # For each cross-word move, find the word bus lane and determine
        # the intermediate site (the site_id where the word bus operates)
        site_adjustments: list[tuple[layout.LocationAddress, layout.LocationAddress]] = []
        word_bus_lanes: dict[int, list[layout.LaneAddress]] = {}

        for src, dst in cross_word:
            # Try direct word bus at dst.site_id first
            adjusted_src = layout.LocationAddress(src.word_id, dst.site_id)
            lane = arch_spec.get_lane_address(adjusted_src, dst)

            if lane is None:
                # dst.site_id doesn't have a word bus — find one that does
                # Try each site in has_word_buses to find a valid word bus
                found = False
                for wb_site in sorted(arch_spec.has_word_buses):
                    wb_src = layout.LocationAddress(src.word_id, wb_site)
                    wb_dst = layout.LocationAddress(dst.word_id, wb_site)
                    lane = arch_spec.get_lane_address(wb_src, wb_dst)
                    if lane is not None:
                        # Site adjustment: src → (src.word, wb_site)
                        if src.site_id != wb_site:
                            site_adjustments.append(
                                (src, layout.LocationAddress(src.word_id, wb_site))
                            )
                        word_bus_lanes.setdefault(lane.bus_id, []).append(lane)
                        # Post word-bus site adjustment: (dst.word, wb_site) → dst
                        # will be handled as a same-word move in a second pass
                        if wb_site != dst.site_id:
                            same_word.append(
                                (layout.LocationAddress(dst.word_id, wb_site), dst)
                            )
                        found = True
                        break
                assert found, f"No word bus path from {src} to {dst}"
            else:
                # Direct word bus at dst.site_id works
                if src.site_id != dst.site_id:
                    site_adjustments.append((src, adjusted_src))
                word_bus_lanes.setdefault(lane.bus_id, []).append(lane)

        # Execute site adjustments first
        if site_adjustments:
            for word_id in {s.word_id for s, _ in site_adjustments}:
                word_diffs = [(s, d) for s, d in site_adjustments if s.word_id == word_id]
                moves.extend(_intra_word_moves(arch_spec, word_diffs))

        # Then word bus moves
        for lanes in word_bus_lanes.values():
            moves.append(tuple(lanes))

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
