"""Shared two-word move synthesis: layout transition to move layers.

Given an architecture spec and two concrete states (before/after layouts),
computes the sequence of move layers for the 2-word logical architecture.
Used by the fixed-home placement strategy and (in the future) by non-fixed
strategies for both right-to-left and left-to-right phases.
"""

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement.lattice import ConcreteState


def _assert_valid_word_bus_move(
    arch_spec: layout.ArchSpec,
    src_word: int,
    src_site: int,
    bus_id: int,
    direction: layout.Direction,
) -> layout.WordLaneAddress:
    lane = layout.WordLaneAddress(
        src_word,
        src_site,
        bus_id,
        direction,
    )
    err = arch_spec.validate_lane(lane)
    assert err == set(), f"Invalid word bus move: {err}"
    return lane


def _assert_valid_site_bus_move(
    arch_spec: layout.ArchSpec,
    src_word: int,
    src_site: int,
    bus_id: int,
    direction: layout.Direction,
) -> layout.SiteLaneAddress:
    lane = layout.SiteLaneAddress(
        src_word,
        src_site,
        bus_id,
        direction,
    )
    err = arch_spec.validate_lane(lane)
    assert err == set(), f"Invalid site bus move: {err}"
    return lane


def _site_moves(
    arch_spec: layout.ArchSpec,
    diffs: list[tuple[layout.LocationAddress, layout.LocationAddress]],
    word_id: int,
) -> list[tuple[layout.LaneAddress, ...]]:
    start_site_ids = [before.site_id for before, _ in diffs]
    assert len(set(start_site_ids)) == len(
        start_site_ids
    ), "Start site ids must be unique"

    bus_moves: dict[int, list[layout.LaneAddress]] = {}
    for before, end in diffs:
        bus_id = (end.site_id % 5) - (before.site_id % 5)
        if bus_id < 0:
            bus_id += len(arch_spec.site_buses)  # wrap around sites
        bus_moves.setdefault(bus_id, []).append(
            _assert_valid_site_bus_move(
                arch_spec,
                word_id,
                before.site_id,
                bus_id,
                layout.Direction.FORWARD,
            )
        )
    return list(map(tuple, bus_moves.values()))


def _compute_move_layers_two_words(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
) -> tuple[tuple[layout.LaneAddress, ...], ...]:
    """Compute move layers for the 2-word logical arch from state_before to state_after."""
    diffs = [
        ele for ele in zip(state_before.layout, state_after.layout) if ele[0] != ele[1]
    ]

    groups: dict[
        tuple[int, int], list[tuple[layout.LocationAddress, layout.LocationAddress]]
    ] = {}
    for src, dst in diffs:
        groups.setdefault((src.word_id, dst.word_id), []).append((src, dst))

    word_moves_10 = groups.get((1, 0), [])
    word_moves_01 = groups.get((0, 1), [])

    assert not (
        word_moves_10 and word_moves_01
    ), "Cannot have both (0,1) and (1,0) moves in logical arch"
    if word_moves_10:
        word_moves = word_moves_10
        word_start = 1
    elif word_moves_01:
        word_moves = word_moves_01
        word_start = 0
    else:
        word_moves = []
        word_start = 0

    moves: list[tuple[layout.LaneAddress, ...]] = _site_moves(
        arch_spec, word_moves, word_start
    )
    # handle word bus moves
    if len(moves) > 0:
        moves.append(
            tuple(
                _assert_valid_word_bus_move(
                    arch_spec,
                    0,
                    end.site_id,
                    0,
                    (
                        layout.Direction.FORWARD
                        if word_start == 0
                        else layout.Direction.BACKWARD
                    ),
                )
                for _, end in word_moves
            )
        )

    # handle site bus moves
    moves.extend(_site_moves(arch_spec, groups.get((0, 0), []), 0))
    moves.extend(_site_moves(arch_spec, groups.get((1, 1), []), 1))

    return tuple(moves)


# Public API: implementation is 2-word only for the time being
compute_move_layers = _compute_move_layers_two_words
