"""Tunable parameters for entropy-guided search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


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

    MIN_DELTA_E: ClassVar[int] = 1
    MIN_E_MAX: ClassVar[int] = 2
    MIN_TOP_C: ClassVar[int] = 1
    MIN_MAX_CANDIDATES: ClassVar[int] = 1
    MIN_REVERSION_STEPS: ClassVar[int] = 1
    MIN_MAX_GOAL_CANDIDATES: ClassVar[int] = 1

    w_d: float = 1.0
    w_m: float = 0.3
    alpha: float = 100.0
    beta: float = 2.0
    gamma: float = 0.5
    top_c: int = 3
    max_candidates: int = 2
    reversion_steps: int = 1
    delta_e: int = 1
    e_max: int = 4
    max_goal_candidates: int = 1

    def __post_init__(self) -> None:
        if self.delta_e < self.MIN_DELTA_E:
            raise ValueError(
                f"delta_e must be >= {self.MIN_DELTA_E} (used as divisor via E_eff)"
            )
        if self.e_max < self.MIN_E_MAX:
            raise ValueError(f"e_max must be >= {self.MIN_E_MAX}")
        if self.top_c < self.MIN_TOP_C:
            raise ValueError(f"top_c must be >= {self.MIN_TOP_C}")
        if self.max_candidates < self.MIN_MAX_CANDIDATES:
            raise ValueError(f"max_candidates must be >= {self.MIN_MAX_CANDIDATES}")
        if self.reversion_steps < self.MIN_REVERSION_STEPS:
            raise ValueError(f"reversion_steps must be >= {self.MIN_REVERSION_STEPS}")
        if self.max_goal_candidates < self.MIN_MAX_GOAL_CANDIDATES:
            raise ValueError(
                f"max_goal_candidates must be >= {self.MIN_MAX_GOAL_CANDIDATES}"
            )
