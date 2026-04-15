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
        w_t: Time-distance blend weight for distance computation.
            0.0 => hop-count only, 1.0 => approximate move-time only.
        alpha: Distance weight in moveset scoring.
        beta: Arrived-gain weight in moveset scoring.
        gamma: Mobility weight in moveset scoring.
        max_candidates: Candidates to try per entropy level before regenerating.
        e_max: Entropy threshold that triggers reversion.
        max_goal_candidates: Number of goal nodes to collect before stopping.
    """

    MIN_E_MAX: ClassVar[int] = 2
    MIN_MAX_CANDIDATES: ClassVar[int] = 1
    MIN_MAX_GOAL_CANDIDATES: ClassVar[int] = 1
    MIN_W_T: ClassVar[float] = 0.0
    MAX_W_T: ClassVar[float] = 1.0

    w_d: float = 0.95
    w_m: float = 0.8
    w_t: float = 0.95
    alpha: float = 80.0
    beta: float = 3.0
    gamma: float = 3.1
    max_candidates: int = 4
    e_max: int = 4
    max_goal_candidates: int = 3

    def __post_init__(self) -> None:
        if self.e_max < self.MIN_E_MAX:
            raise ValueError(f"e_max must be >= {self.MIN_E_MAX}")
        if self.max_candidates < self.MIN_MAX_CANDIDATES:
            raise ValueError(f"max_candidates must be >= {self.MIN_MAX_CANDIDATES}")
        if self.max_goal_candidates < self.MIN_MAX_GOAL_CANDIDATES:
            raise ValueError(
                f"max_goal_candidates must be >= {self.MIN_MAX_GOAL_CANDIDATES}"
            )
        if not (self.MIN_W_T <= self.w_t <= self.MAX_W_T):
            raise ValueError(f"w_t must be in [{self.MIN_W_T}, {self.MAX_W_T}]")
