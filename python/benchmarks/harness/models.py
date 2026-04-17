"""Shared data models for benchmark harness execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from kirin import ir

from bloqade.lanes.analysis.placement import PlacementStrategyABC

Backend = Literal["python", "rust"]


@dataclass(frozen=True)
class BenchmarkCase:
    """Reproducible benchmark input circuit."""

    case_id: str
    kernel: ir.Method
    tags: tuple[str, ...] = ()
    logical_initialize: bool = True


@dataclass(frozen=True)
class StrategyConfig:
    """One benchmark strategy row in the case-by-strategy matrix."""

    strategy_id: str
    backend: Backend
    generator_id: str
    build_placement_strategy: Callable[[], PlacementStrategyABC]
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkJob:
    """Expanded benchmark job: one case under one strategy."""

    case: BenchmarkCase
    strategy: StrategyConfig


@dataclass(frozen=True)
class BenchmarkRow:
    """One output record in CSV/console outputs."""

    case_id: str
    strategy_id: str
    backend: Backend
    generator_id: str
    success: bool
    wall_time_ms: float | None
    move_count_events: int | None
    move_count_lanes: int | None
    estimated_fidelity: float | None
    nodes_explored: int | None
    max_depth_reached: int | None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
