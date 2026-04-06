"""Tunable parameters for entropy-guided search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchParams:
    """Tunable parameters for entropy-guided search.

    Attributes:
        w_d: Distance weight in per-qubit lane scoring.
        w_m: Mobility weight in per-qubit lane scoring.
        alpha: Distance weight in moveset scoring.
        beta: Arrived-gain weight in moveset scoring.
        gamma: Mobility weight in moveset scoring.
        top_c: Legacy per-qubit shortlist width (unused by rectangle heuristic).
        max_candidates: Candidates to try per entropy level before regenerating.
        reversion_steps: Steps to revert up the tree on deadlock.
        delta_e: Entropy increment per revisit or failed generation.
        e_max: Entropy threshold that triggers reversion.
        max_goal_candidates: Number of goal nodes to collect before stopping.
    """

    w_d: float = 1.0
    w_m: float = 0.1
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.5
    top_c: int = 3
    max_candidates: int = 2
    reversion_steps: int = 1
    delta_e: int = 1
    e_max: int = 4
    max_goal_candidates: int = 1

    def __post_init__(self) -> None:
        if self.delta_e < 1:
            raise ValueError("delta_e must be >= 1 (used as divisor via E_eff)")
        if self.e_max < 2:
            raise ValueError("e_max must be >= 2")
        if self.top_c < 1:
            raise ValueError("top_c must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.reversion_steps < 1:
            raise ValueError("reversion_steps must be >= 1")
        if self.max_goal_candidates < 1:
            raise ValueError("max_goal_candidates must be >= 1")
