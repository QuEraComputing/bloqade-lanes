"""No-return placement strategy using the loose-goal entangling solver.

Instead of fixed target positions, this strategy passes CZ pair constraints
to the Rust ``solve_entangling`` solver, which simultaneously discovers
both the entangling placement and the routing. Layers are chained: the
output configuration of one CZ layer becomes the input for the next, so
atoms do not return to their home positions between CZ gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import (
    LooseGoalCzPlacement,
    MoveSearch,
    SearchEngine,
    SolveResult,
)
from bloqade.lanes.heuristics.physical._no_return_base import NoReturnStrategyBase


@dataclass
class NoReturnPlacementStrategy(NoReturnStrategyBase):
    """No-return placement via the loose-goal entangling constraint solver.

    Calls :class:`LooseGoalCzPlacement` once per CZ layer to find both the
    entangling placement and the routing simultaneously. Each layer's output
    layout is passed as the next layer's input, saving the cost of palindrome
    return moves.

    Parameters
    ----------
    arch_spec:
        Architecture specification.
    strategy:
        Inner search strategy as a :class:`SearchStrategy` enum (e.g.
        :py:attr:`SearchStrategy.IDS` (default),
        :py:attr:`SearchStrategy.ASTAR`, :py:attr:`SearchStrategy.ENTROPY`).
    max_expansions:
        Maximum node expansions per solve call.
    restarts:
        Number of parallel restarts with perturbed scoring. Each restart
        gets a different seed for the greedy CZ-pair-to-slot assignment,
        producing diverse target layouts; ``pick_best`` keeps the lowest-
        cost result. Default ``20``.
    deadlock_policy:
        :class:`DeadlockPolicy` enum value (default
        :py:attr:`DeadlockPolicy.MOVE_BLOCKERS`).
    top_c:
        Per-qubit move-candidate pruning cap inside ``HeuristicGenerator``.
        ``None`` keeps all scored bus options. Default ``3`` matches the
        previously-hardcoded behaviour. Larger values broaden the search
        but slow per-node expansion.
    congestion_weight:
        Penalty weight for the entangling Hungarian assignment to spread
        CZ pairs across word pairs. ``0.0`` (default) uses standard
        min-sum assignment; positive values reduce routing serialization
        at high occupancy at some cost in total atom moves.
    occupancy_penalty:
        Per-slot-half penalty (in lane-hop units) added to the Hungarian
        cost for slots currently held by spectator atoms (atoms not in any
        CZ pair of the current layer). Steers the assignment away from
        slots that would force the search to evict a non-participating
        atom. ``0.0`` recovers the legacy occupancy-blind behaviour.
        Default ``1.0`` was tuned on the 80q / depth 3 / max_pairs 10
        regime; deeper sparse-pair circuits prefer larger values (~2–3).
    hungarian_horizon:
        Cap on the number of future CZ layers fed to the Hungarian
        forward/backward sweep. ``0`` disables lookahead entirely; ``None``
        is unbounded (all future layers). Default ``4`` keeps solve time
        bounded regardless of circuit depth.
    """

    restarts: int = 20
    top_c: int | None = 3
    congestion_weight: float = 0.0
    occupancy_penalty: float = 1.0
    hungarian_horizon: int | None = 4

    def _build_entangling_options(self) -> _native.EntanglingOptions:
        return _native.EntanglingOptions(
            congestion_weight=self.congestion_weight,
            occupancy_penalty=self.occupancy_penalty,
            hungarian_horizon=self.hungarian_horizon,
        )

    def _invoke_placement(
        self,
        engine: SearchEngine,
        move_search: MoveSearch,
        initial: dict[int, "_native.LocationAddress"],
        cz_pairs: list[tuple[int, int]],
        blocked: list["_native.LocationAddress"],
        future_cz_layers: list[list[tuple[int, int]]] | None,
    ) -> SolveResult:
        placement = LooseGoalCzPlacement(
            engine, move_search, self._build_entangling_options()
        )
        return placement.solve_pairs(
            initial,
            cz_pairs,
            blocked,
            max_expansions=self.max_expansions,
            future_cz_layers=future_cz_layers,
        )
