"""Smoke test for the migrated EntropyScorer signature.

Verifies that `EntropyScorer.metrics()` and `.score_moveset()` accept
`list[LaneAddress]` (post-migration) and return `MovesetMetrics` with the
expected attribute surface. Without this, no other test exercises the
binding's `Vec<PyRef<PyLaneAddr>>` signature.
"""

from __future__ import annotations

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode._native import (
    EntropyScorer,
    LocationAddress as NativeLocationAddress,
    MovesetMetrics,
)


def test_entropy_scorer_constructs_and_scores_empty_moveset():
    """Empty moveset is a no-op; exercises the signature and constructor path."""
    arch_spec = logical.get_arch_spec()
    target = {
        0: NativeLocationAddress(0, 0, 0),
        1: NativeLocationAddress(0, 1, 0),
    }
    current = {
        0: NativeLocationAddress(0, 0, 0),
        1: NativeLocationAddress(0, 1, 0),
    }

    scorer = EntropyScorer(arch_spec._inner, target)
    metrics = scorer.metrics(current, [])

    assert isinstance(metrics, MovesetMetrics)
    # Empty moveset => no progress, no arrivals, no mobility delta.
    assert metrics.distance_progress == 0.0
    assert metrics.arrived == 0
    assert metrics.closer == []
    assert metrics.further == []


def test_entropy_scorer_score_moveset_returns_float():
    arch_spec = logical.get_arch_spec()
    target = {0: NativeLocationAddress(0, 0, 0)}
    current = {0: NativeLocationAddress(0, 0, 0)}

    scorer = EntropyScorer(arch_spec._inner, target)
    score = scorer.score_moveset(current, [])

    assert isinstance(score, float)


def test_entropy_scorer_metrics_accepts_lane_address_moveset():
    """Pass an actual LaneAddress through the binding to exercise PyO3 extraction.

    Uses lanes harvested from a real solve so they're guaranteed to exist
    in the arch's lane index.
    """
    from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
    from bloqade.lanes.bytecode.encoding import LocationAddress
    from bloqade.lanes.heuristics.physical.movement import (
        PhysicalPlacementStrategy,
        RustPlacementTraversal,
    )

    arch_spec = logical.get_arch_spec()
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec, traversal=RustPlacementTraversal()
    )
    state = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
        move_count=(0, 0),
    )
    out = strategy.cz_placements(state, controls=(0,), targets=(1,))
    assert isinstance(out, ExecuteCZ)

    # If the initial state already entangles, move_layers is empty — but the
    # scorer call still verifies the binding works with that empty list.
    moveset = list(out.move_layers[0]) if out.move_layers else []
    inner_moveset = [lane._inner for lane in moveset]

    target = {0: out.layout[0]._inner, 1: out.layout[1]._inner}
    current = {0: state.layout[0]._inner, 1: state.layout[1]._inner}

    scorer = EntropyScorer(arch_spec._inner, target)
    metrics = scorer.metrics(current, inner_moveset)

    assert isinstance(metrics, MovesetMetrics)
    assert hasattr(metrics, "score")
