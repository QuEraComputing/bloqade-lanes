"""No-home placement strategy: two-phase return assignment + entangling routing.

Instead of returning atoms to their original home positions after each CZ
layer, this strategy assigns displaced qubits to *optimal* home sites that
minimise future movement.  The assignment uses the Hungarian algorithm with
gamma-decayed future CZ partner proximity as a lookahead signal.

Phase 1 (return): Hungarian-pick a home layout, then route current → home
via fixed-target ``solve``.
Phase 2 (entangling): Hungarian-pick CZ-staging targets (with optional
lookahead-aware blend), then route home → staging via fixed-target ``solve``.
This mirrors how :class:`PhysicalPlacementStrategy` routes to pre-computed
CZ targets.

Both phases run in Rust via ``MoveSolver.solve_nohome``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    PlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import MoveSolver
from bloqade.lanes.layout import (
    LaneAddress,
    LocationAddress,
    ZoneAddress,
)


@dataclass
class NoHomePlacementStrategy(PlacementStrategyABC):
    """Two-phase placement: return assignment + entangling routing.

    Parameters
    ----------
    arch_spec:
        Architecture specification.
    strategy:
        Search strategy name for the routing phases.
    max_expansions:
        Maximum node expansions per solve call (shared across phases).
    restarts:
        Number of parallel restarts with perturbed scoring inside each
        routing solve (return phase + entangling phase). The two-phase
        Hungarian assignments themselves are deterministic; diversity
        comes from the candidate-home-layout fan-out.
    lookahead:
        Enable 2-step lookahead scoring in the routing search.
    deadlock_policy:
        How the heuristic generator handles deadlocks during routing.
    gamma:
        Discount factor for future CZ layer weights in the return
        assignment (default 0.85).
    lambda_lookahead:
        Blend weight for future proximity penalty in the return
        assignment (default 0.5).
    k_candidates:
        Maximum candidate holes per returner for cost-matrix pruning
        (default 8).
    """

    strategy: str = "ids"
    max_expansions: int | None = 100
    restarts: int = 20
    lookahead: bool = True
    deadlock_policy: str = "move_blockers"  # "skip" | "move_blockers" | "all_moves"
    gamma: float = 0.85
    lambda_lookahead: float = 0.5
    k_candidates: int = 8

    _solver: MoveSolver | None = field(default=None, init=False, repr=False)

    _STRATEGY_MAP: dict[str, _native.SearchStrategy] = field(
        default_factory=lambda: {
            "ids": _native.SearchStrategy.IDS,
            "cascade": _native.SearchStrategy.CASCADE_IDS,
            "cascade-ids": _native.SearchStrategy.CASCADE_IDS,
            "entropy": _native.SearchStrategy.ENTROPY,
            "astar": _native.SearchStrategy.ASTAR,
        },
        init=False,
        repr=False,
    )

    _DEADLOCK_MAP: dict[str, _native.DeadlockPolicy] = field(
        default_factory=lambda: {
            "skip": _native.DeadlockPolicy.SKIP,
            "move_blockers": _native.DeadlockPolicy.MOVE_BLOCKERS,
            "all_moves": _native.DeadlockPolicy.ALL_MOVES,
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)

    def _get_solver(self) -> MoveSolver:
        if self._solver is None:
            self._solver = MoveSolver.from_arch_spec(self.arch_spec._inner)
        return self._solver

    def _make_options(self) -> _native.SolveOptions:
        return _native.SolveOptions(
            strategy=self._STRATEGY_MAP[self.strategy],
            restarts=self.restarts,
            lookahead=self.lookahead,
            deadlock_policy=self._DEADLOCK_MAP[self.deadlock_policy],
        )

    def _make_nohome_options(self) -> _native.NoHomeOptions:
        return _native.NoHomeOptions(
            gamma=self.gamma,
            lambda_lookahead=self.lambda_lookahead,
            k_candidates=self.k_candidates,
        )

    def validate_initial_layout(
        self, initial_layout: tuple[LocationAddress, ...]
    ) -> None:
        pass

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()

        if not isinstance(state, ConcreteState):
            return AtomState.top()

        solver = self._get_solver()
        options = self._make_options()
        nohome_options = self._make_nohome_options()

        # Build initial placement: qubit_id -> native location.
        initial = {qid: state.layout[qid]._inner for qid in range(len(state.layout))}

        # Build CZ pairs from controls/targets.
        cz_pairs = list(zip(controls, targets))

        # Blocked locations: occupied by atoms not in this circuit.
        blocked = [loc._inner for loc in state.occupied]

        # Extract future CZ layers for lookahead-aware assignment.
        # lookahead_cz_layers[0] is the current layer (skip it),
        # lookahead_cz_layers[1:] are future layers.
        future = None
        if len(lookahead_cz_layers) > 1:
            future = [list(zip(ctrls, tgts)) for ctrls, tgts in lookahead_cz_layers[1:]]

        result = solver.solve_nohome(
            initial,
            cz_pairs,
            blocked,
            max_expansions=self.max_expansions,
            options=options,
            nohome_options=nohome_options,
            future_cz_layers=future,
        )

        if result.status != "solved":
            return AtomState.bottom()

        # Convert move layers to LaneAddress tuples.
        move_layers = tuple(
            tuple(LaneAddress.from_inner(lane) for lane in step)
            for step in result.move_layers
        )

        # Build goal layout from solver result.
        goal_map = {
            qid: LocationAddress(loc.word_id, loc.site_id, loc.zone_id)
            for qid, loc in result.goal_config.items()
        }
        goal_layout = tuple(goal_map[qid] for qid in range(len(state.layout)))

        move_count = tuple(
            mc + int(src != dst)
            for mc, src, dst in zip(state.move_count, state.layout, goal_layout)
        )

        return ExecuteCZ(
            occupied=state.occupied,
            layout=goal_layout,
            move_count=move_count,
            active_cz_zones=self.arch_spec.cz_zone_addresses,
            move_layers=move_layers,
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        if isinstance(state, ConcreteState):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state

        if len(qubits) != len(state.layout):
            return AtomState.bottom()

        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(ZoneAddress(loc.zone_id) for loc in state.layout),
        )
