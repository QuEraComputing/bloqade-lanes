"""Move synthesis: layout transition to move layers.

Given an architecture spec and two concrete states (before/after layouts),
computes the sequence of move layers using the Rust MoveSolver.
"""

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement.lattice import ConcreteState
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import MoveSolver
from bloqade.lanes.heuristics.physical.movement import convert_move_layers
from bloqade.lanes.layout.path import PathFinder


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


def compute_move_layers(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
    solver: MoveSolver | None = None,
) -> tuple[tuple[layout.LaneAddress, ...], ...]:
    """Compute move layers from state_before to state_after via the Rust MoveSolver.

    If ``solver`` is provided, it is reused instead of constructing a fresh
    ``MoveSolver`` — callers that invoke this repeatedly with the same
    ``arch_spec`` should cache a solver and pass it in.
    """
    initial_native = {qid: loc._inner for qid, loc in enumerate(state_before.layout)}
    target_native = {qid: loc._inner for qid, loc in enumerate(state_after.layout)}
    blocked_native = [loc._inner for loc in state_before.occupied]

    if solver is None:
        solver = MoveSolver.from_arch_spec(arch_spec._inner)
    opts = _native.SolveOptions(
        strategy=_native.SearchStrategy.ENTROPY,
        max_movesets_per_group=3,
        max_goal_candidates=3,
        collect_entropy_trace=False,
    )
    result = solver.solve(
        initial_native,
        target_native,
        blocked_native,
        max_expansions=None,
        options=opts,
    )
    if result.status != "solved":
        raise RuntimeError(f"move synthesis failed with status={result.status!r}")
    return convert_move_layers(result.move_layers)


def move_to_entangle(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
    solver: MoveSolver | None = None,
) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...]]:
    """Synthesize move layers from current layout to CZ entangling layout."""
    return state_after, compute_move_layers(
        arch_spec, state_before, state_after, solver=solver
    )


def move_to_left(
    arch_spec: layout.ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
    solver: MoveSolver | None = None,
) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...]]:
    """Synthesize move layers from CZ layout to post-CZ return layout."""
    forward_layers = compute_move_layers(
        arch_spec, state_after, state_before, solver=solver
    )
    reverse_layers = tuple(
        tuple(lane.reverse() for lane in move_lanes[::-1])
        for move_lanes in forward_layers[::-1]
    )
    return state_after, reverse_layers
