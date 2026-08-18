"""Shared data models for benchmark harness execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from kirin import ir

from bloqade.lanes.analysis.placement import PlacementStrategyABC

Backend = Literal["python", "rust"]

BUILTIN_ARCH_SPEC_ID = "builtin"


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
    arch_spec_id: str = BUILTIN_ARCH_SPEC_ID
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
    cuts_by_g: int | None = None
    """Branches pruned because accumulated cost alone reached the incumbent."""
    cuts_by_h: int | None = None
    """Branches pruned only because the completion bound was added: the bound's
    actual contribution over the pre-existing test."""
    cuts_infeasible: int | None = None
    """Branches pruned by an ``h = +inf`` proof that no completion exists,
    independent of any incumbent. Reported separately because a solve whose
    pruning is entirely infeasibility proofs leaves the other counters at zero,
    which would otherwise be indistinguishable from the bound doing nothing."""
    cut_depth_sum: int | None = None
    """Summed depth at which ``cuts_by_h`` fired."""
    cut_depth_g_only_sum: int | None = None
    """Summed depth at which cost alone would have reached the incumbent for
    those same branches. Against ``cut_depth_sum`` this is the depth ratio:
    how much earlier the bound cut, hence roughly how much subtree it saved."""
    max_optimality_gap: float | None = None
    """Worst ``(incumbent - h(root)) / incumbent`` over this case's solves.
    ``0.0`` means provably optimal; larger positive values mean the bound is
    loose and a tighter one has headroom.

    A **negative** value is out of band: it means ``h(root) > incumbent``, i.e.
    an inadmissible bound that may have pruned the optimum. Rust preserves the
    sign deliberately, and the accumulator lets a negative reading dominate any
    non-negative one rather than aggregating it away — so this field is a "worst
    gap" where a violated admissibility invariant counts as worse than any loose
    bound, not a plain maximum."""
    arch_spec_id: str = BUILTIN_ARCH_SPEC_ID
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
