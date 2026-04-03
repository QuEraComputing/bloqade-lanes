"""Structured metadata for entropy-guided search on_step callbacks."""

from __future__ import annotations

__all__ = [
    "StepInfo",
    "GoalStepInfo",
    "FallbackStartStepInfo",
    "FallbackStepInfo",
    "RevertStepInfo",
    "EntropyBumpStepInfo",
    "DescendStepInfo",
]

from dataclasses import dataclass, field

from bloqade.lanes.layout import LaneAddress


@dataclass
class StepInfo:
    """Base metadata passed to on_step callbacks."""

    entropy: int
    unresolved_count: int
    candidate_movesets: tuple[frozenset[LaneAddress], ...] = ()
    candidate_index: int | None = None


@dataclass
class GoalStepInfo(StepInfo):
    """Metadata emitted when the goal configuration is reached."""

    total_depth: int = 0


@dataclass
class FallbackStartStepInfo(StepInfo):
    """Metadata emitted when the sequential fallback begins."""

    unresolved_qubits: list[int] = field(default_factory=list)


@dataclass
class FallbackStepInfo(StepInfo):
    """Metadata emitted for each step of the sequential fallback."""

    qubit_id: int | None = None
    moveset: frozenset[LaneAddress] = field(default_factory=frozenset)


@dataclass
class RevertStepInfo(StepInfo):
    """Metadata emitted when the search reverts to an ancestor node."""

    reversion_steps: int = 0
    ancestor_depth: int = 0
    reason: str = ""
    state_seen_node_id: int | None = None
    no_valid_moves_qubit: int | None = None
    trigger_node_id: int | None = None
    trigger_entropy: int | None = None


@dataclass
class EntropyBumpStepInfo(StepInfo):
    """Metadata emitted when a node's entropy is incremented."""

    new_entropy: int = 0
    reason: str = ""
    no_valid_moves_qubit: int | None = None
    state_seen_node_id: int | None = None
    attempted_moveset: frozenset[LaneAddress] = field(default_factory=frozenset)


@dataclass
class DescendStepInfo(StepInfo):
    """Metadata emitted when the search descends to a new child node."""

    moveset: frozenset[LaneAddress] = field(default_factory=frozenset)
    moveset_score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
