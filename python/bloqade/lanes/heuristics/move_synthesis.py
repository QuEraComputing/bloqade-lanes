"""Move synthesis: layout transition to move layers.

Given an architecture spec and two concrete states (before/after layouts),
computes the sequence of move layers using the Rust MoveSolver.
"""

from bloqade.lanes.analysis.placement.lattice import ConcreteState
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import MoveSolver
from bloqade.lanes.bytecode.encoding import LaneAddress
from bloqade.lanes.heuristics.physical.movement import (
    RustPlacementTraversal,
    convert_move_layers,
    solve_options_from_traversal,
)

_DEFAULT_TRAVERSAL = RustPlacementTraversal()


def compute_move_layers(
    arch_spec: ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
    solver: MoveSolver | None = None,
    traversal: RustPlacementTraversal = _DEFAULT_TRAVERSAL,
) -> tuple[tuple[LaneAddress, ...], ...]:
    """Compute move layers from state_before to state_after via the Rust MoveSolver.

    If ``solver`` is provided, it is reused instead of constructing a fresh
    ``MoveSolver`` — callers that invoke this repeatedly with the same
    ``arch_spec`` should cache a solver and pass it in. ``traversal`` selects
    the search strategy and bounds; it shares ``RustPlacementTraversal``'s
    defaults with ``PhysicalPlacementStrategy`` so the two callsites cannot
    drift.
    """
    initial_native = {qid: loc._inner for qid, loc in enumerate(state_before.layout)}
    target_native = {qid: loc._inner for qid, loc in enumerate(state_after.layout)}
    blocked_native = [loc._inner for loc in state_before.occupied]

    if solver is None:
        solver = MoveSolver.from_arch_spec(arch_spec._inner)
    result = solver.solve(
        initial_native,
        target_native,
        blocked_native,
        max_expansions=None,
        options=solve_options_from_traversal(traversal),
    )
    if result.status != "solved":
        raise RuntimeError(f"move synthesis failed with status={result.status!r}")
    return convert_move_layers(result.move_layers)


def move_to_entangle(
    arch_spec: ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
    solver: MoveSolver | None = None,
) -> tuple[ConcreteState, tuple[tuple[LaneAddress, ...], ...]]:
    """Synthesize move layers from current layout to CZ entangling layout."""
    return state_after, compute_move_layers(
        arch_spec, state_before, state_after, solver=solver
    )


def move_to_left(
    arch_spec: ArchSpec,
    state_before: ConcreteState,
    state_after: ConcreteState,
    solver: MoveSolver | None = None,
) -> tuple[ConcreteState, tuple[tuple[LaneAddress, ...], ...]]:
    """Synthesize move layers from CZ layout to post-CZ return layout."""
    forward_layers = compute_move_layers(
        arch_spec, state_after, state_before, solver=solver
    )
    reverse_layers = tuple(
        tuple(lane.reverse() for lane in move_lanes[::-1])
        for move_lanes in forward_layers[::-1]
    )
    return state_after, reverse_layers
