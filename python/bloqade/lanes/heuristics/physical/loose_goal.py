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
from bloqade.lanes.layout import (
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
        Number of parallel restarts with perturbed scoring.
    dynamic_targets:
        Whether to recompute per-qubit targets dynamically during search.
    recompute_interval:
        How often to recompute targets when ``dynamic_targets`` is true.
        0 = deadlock-triggered only.
    """

    strategy: str = "ids"
    max_expansions: int | None = 100
    restarts: int = 20
    dynamic_targets: bool = True
    recompute_interval: int = 0
    lookahead: bool = True

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

    def _make_options(self) -> _native.SolveOptions:
        return _native.SolveOptions(
            strategy=self._STRATEGY_MAP[self.strategy],
            restarts=self.restarts,
            dynamic_targets=self.dynamic_targets,
            recompute_interval=self.recompute_interval,
            lookahead=self.lookahead,
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

        # Build initial placement: qubit_id → native location.
        initial = {qid: state.layout[qid]._inner for qid in range(len(state.layout))}

        # Build CZ pairs from controls/targets.
        cz_pairs = list(zip(controls, targets))

        # Blocked locations: occupied by atoms not in this circuit.
        blocked = [loc._inner for loc in state.occupied]

        result = solver.solve_entangling(
            initial,
            cz_pairs,
            blocked,
            max_expansions=self.max_expansions,
            options=options,
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
