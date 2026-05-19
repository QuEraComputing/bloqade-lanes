"""No-home placement strategy: two-phase return assignment + entangling routing.

Instead of returning atoms to their original home positions after each CZ
layer, this strategy assigns displaced qubits to *optimal* home sites that
minimise future movement. The assignment uses the Hungarian algorithm with
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

from dataclasses import dataclass

from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import MoveSolver, SolveResult
from bloqade.lanes.heuristics.physical._no_return_base import NoReturnStrategyBase


@dataclass
class NoHomePlacementStrategy(NoReturnStrategyBase):
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
        comes from the candidate-home-layout fan-out. Default ``20``.
    deadlock_policy:
        How the heuristic generator handles deadlocks during routing.
    gamma:
        Discount factor for future CZ layer weights in the return
        assignment (default ``0.85``).
    lambda_lookahead:
        Blend weight for future proximity penalty in the return
        assignment (default ``0.5``).
    k_candidates:
        Maximum candidate holes per returner for cost-matrix pruning
        (default ``8``).
    top_bus_signatures:
        Number of bus-reward variant assignments to generate (default
        ``6``). Each variant rewards edges sharing a high-coverage lane
        signature, biasing the assignment toward layouts with parallel
        routing.
    bus_reward_rho:
        Per-edge hop-count discount applied to edges using a top
        signature when building bus-reward variant cost matrices
        (default ``1``).

    Notes
    -----
    Unlike :class:`NoReturnPlacementStrategy` and
    :class:`RecedingHorizonNoReturnPlacementStrategy`, this strategy does
    not pass ``top_c`` to ``SolveOptions`` (it inherits the base default
    ``None``), matching the historical behaviour of the two-phase
    ``solve_nohome`` entry point.
    """

    restarts: int = 20
    gamma: float = 0.85
    lambda_lookahead: float = 0.5
    k_candidates: int = 8
    top_bus_signatures: int = 6
    bus_reward_rho: int = 1

    def _build_nohome_options(self) -> _native.NoHomeOptions:
        return _native.NoHomeOptions(
            gamma=self.gamma,
            lambda_lookahead=self.lambda_lookahead,
            k_candidates=self.k_candidates,
            top_bus_signatures=self.top_bus_signatures,
            bus_reward_rho=self.bus_reward_rho,
        )

    def _invoke_solver(
        self,
        solver: MoveSolver,
        initial: dict[int, "_native.LocationAddress"],
        cz_pairs: list[tuple[int, int]],
        blocked: list["_native.LocationAddress"],
        future_cz_layers: list[list[tuple[int, int]]] | None,
    ) -> SolveResult:
        return solver.solve_nohome(
            initial,
            cz_pairs,
            blocked,
            max_expansions=self.max_expansions,
            options=self._build_solve_options(),
            nohome_options=self._build_nohome_options(),
            future_cz_layers=future_cz_layers,
        )
