"""Loose-goal placement strategy using entangling constraint search.

Instead of fixed target positions, this strategy passes CZ pair constraints
to the Rust ``solve_entangling`` solver which simultaneously discovers both
the entangling placement and the routing.  Layers are chained: the output
configuration of one CZ layer becomes the input for the next, avoiding the
cost of returning to home positions between layers.
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
from bloqade.lanes.bytecode.encoding import (
    LaneAddress,
    LocationAddress,
    ZoneAddress,
)


@dataclass
class LooseGoalPlacementStrategy(PlacementStrategyABC):
    """Placement strategy using loose-goal entangling constraint search.

    Uses :pymethod:`MoveSolver.solve_entangling` to find both the entangling
    placement and routing simultaneously.  Each CZ layer's output layout is
    passed as input to the next layer (chaining), saving the cost of
    palindrome return moves.

    Parameters
    ----------
    arch_spec:
        Architecture specification.
    strategy:
        Search strategy name (``"ids"``, ``"cascade"``, ``"entropy"``, etc.).
    max_expansions:
        Maximum node expansions per solve call.
    restarts:
        Number of parallel restarts with perturbed scoring.  Each restart
        gets a different seed for the greedy CZ-pair-to-slot assignment,
        producing diverse target layouts; ``pick_best`` keeps the lowest-
        cost result.
    congestion_weight:
        Penalty weight for the entangling Hungarian assignment to spread
        CZ pairs across word pairs. ``0.0`` (default) uses standard
        min-sum assignment; positive values reduce routing serialization
        at high occupancy at some cost in total atom moves.
    occupancy_penalty:
        Per-slot-half penalty (in lane-hop units) added to the Hungarian
        cost for slots currently held by spectator atoms (atoms not in
        any CZ pair of the current layer). Steers the assignment away
        from slots that would force the search to evict a
        non-participating atom. ``0.0`` recovers the legacy
        occupancy-blind behaviour. Default ``1.0`` was tuned on the 80q
        / depth 3 / max_pairs 10 regime; deeper sparse-pair circuits
        prefer larger values (~2–3). Fractional values are supported.
    hungarian_horizon:
        Cap on the number of future CZ layers fed to the Hungarian
        forward/backward sweep. ``0`` disables lookahead entirely;
        ``None`` is unbounded (all future layers). Each extra layer
        costs an extra Hungarian pass per restart, so solve time grows
        linearly in horizon. Default ``4`` keeps solve time bounded
        regardless of circuit depth.
    top_c:
        Per-qubit move-candidate pruning cap inside ``HeuristicGenerator``.
        ``None`` keeps all scored bus options. Default ``3`` matches the
        previously-hardcoded behaviour. Larger values broaden the search
        but slow per-node expansion.
    deadlock_policy:
        ``"skip"`` | ``"move_blockers"`` (default) | ``"all_moves"``.
    """

    strategy: str = "ids"
    max_expansions: int | None = 100
    restarts: int = 20
    congestion_weight: float = 0.0
    occupancy_penalty: float = 1.0
    hungarian_horizon: int | None = 4
    top_c: int | None = 3
    deadlock_policy: str = "move_blockers"

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

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)

    def _get_solver(self) -> MoveSolver:
        if self._solver is None:
            self._solver = MoveSolver.from_arch_spec(self.arch_spec._inner)
        return self._solver

    _DEADLOCK_MAP: dict[str, _native.DeadlockPolicy] = field(
        default_factory=lambda: {
            "skip": _native.DeadlockPolicy.SKIP,
            "move_blockers": _native.DeadlockPolicy.MOVE_BLOCKERS,
            "all_moves": _native.DeadlockPolicy.ALL_MOVES,
        },
        init=False,
        repr=False,
    )

    def _make_options(self) -> tuple[_native.SolveOptions, _native.EntanglingOptions]:
        opts = _native.SolveOptions(
            strategy=self._STRATEGY_MAP[self.strategy],
            restarts=self.restarts,
            lookahead=True,
            deadlock_policy=self._DEADLOCK_MAP[self.deadlock_policy],
            top_c=self.top_c,
        )
        ent_opts = _native.EntanglingOptions(
            congestion_weight=self.congestion_weight,
            occupancy_penalty=self.occupancy_penalty,
            hungarian_horizon=self.hungarian_horizon,
        )
        return opts, ent_opts

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
        options, entangling_options = self._make_options()

        # Build initial placement: qubit_id → native location.
        initial = {qid: state.layout[qid]._inner for qid in range(len(state.layout))}

        # Build CZ pairs from controls/targets.
        cz_pairs = list(zip(controls, targets))

        # Blocked locations: occupied by atoms not in this circuit.
        blocked = [loc._inner for loc in state.occupied]

        # Forward all future CZ layers to Rust unclipped — the Hungarian
        # horizon clip lives Rust-side via ``EntanglingOptions.hungarian_horizon``.
        # lookahead_cz_layers[0] is the current layer (skip it).
        future = None
        if len(lookahead_cz_layers) > 1:
            future = [list(zip(ctrls, tgts)) for ctrls, tgts in lookahead_cz_layers[1:]]

        result = solver.solve_entangling(
            initial,
            cz_pairs,
            blocked,
            max_expansions=self.max_expansions,
            options=options,
            entangling_options=entangling_options,
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
