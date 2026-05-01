"""Tests for LookaheadCongestionAwareTargetGenerator."""

from __future__ import annotations

import pytest

from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
from bloqade.lanes.arch.gemini.physical import (
    get_arch_spec as get_physical_arch_spec,
)
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.movement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)
from bloqade.lanes.heuristics.physical.target_generator import (
    DefaultTargetGenerator,
    LookaheadCongestionAwareTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
)

# ---------------------------------------------------------------------- #
# 1. Protocol conformance + constructor validation                        #
# ---------------------------------------------------------------------- #


def test_implements_target_generator_protocol():
    gen = LookaheadCongestionAwareTargetGenerator()
    assert isinstance(gen, TargetGeneratorABC)


def test_constructor_default_arguments():
    gen = LookaheadCongestionAwareTargetGenerator()
    assert gen.K == 4
    assert gen.gamma == 0.7
    assert gen.direction_factor == 0.5
    assert gen.shared_site_factor == 1.1


@pytest.mark.parametrize("bad_K", [-1, -10])
def test_rejects_negative_K(bad_K):
    with pytest.raises(ValueError, match=r"K=-?\d+ must be >= 0"):
        LookaheadCongestionAwareTargetGenerator(K=bad_K)


@pytest.mark.parametrize("bad_gamma", [0.0, -0.1, 1.5])
def test_rejects_invalid_gamma(bad_gamma):
    with pytest.raises(ValueError, match=r"gamma=.* must be in \(0, 1\]"):
        LookaheadCongestionAwareTargetGenerator(gamma=bad_gamma)


def test_rejects_nonpositive_direction_factor():
    # Inherited validation from CongestionAwareTargetGenerator.
    with pytest.raises(ValueError, match=r"direction_factor=.* must be"):
        LookaheadCongestionAwareTargetGenerator(direction_factor=0.0)


def test_rejects_negative_shared_site_factor():
    # Inherited validation from CongestionAwareTargetGenerator.
    with pytest.raises(ValueError, match=r"shared_site_factor=.* must be"):
        LookaheadCongestionAwareTargetGenerator(shared_site_factor=-0.1)


# ---------------------------------------------------------------------- #
# 2. K=0 behaves like CongestionAware (no future contribution)            #
# ---------------------------------------------------------------------- #


def test_K0_returns_single_candidate():
    """With K=0 the simulated future cost is zero; output should be
    a single congestion-aware candidate, identical in structure to
    CongestionAwareTargetGenerator's."""
    arch = get_physical_arch_spec()
    qubits = (0, 1, 2, 3)
    stages = [((0, 1), (2, 3))]
    layout = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=arch
    ).compute_layout(qubits, stages)
    state = ConcreteState(
        occupied=frozenset(),
        layout=tuple(layout),
        move_count=tuple(0 for _ in layout),
    )
    ctx = TargetContext(
        arch_spec=arch,
        state=state,
        controls=(0, 2),
        targets=(1, 3),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )
    gen = LookaheadCongestionAwareTargetGenerator(K=0)
    cands = list(gen.generate(ctx))
    assert len(cands) == 1


# ---------------------------------------------------------------------- #
# 3. Behavioural — multi-hub-multi-spoke (the canonical sweet-spot)       #
# ---------------------------------------------------------------------- #


def _hub_swap_chain(n_hubs, spokes_per_hub, n_rounds):
    """H hubs each interact with their own n_spokes-spoke sequence
    over R rounds. This is the empirically strongest sweet spot for
    LookaheadCongestionAware (1.20-1.44× lane reduction at H>=3)."""
    qubits = tuple(range(n_hubs + n_hubs * spokes_per_hub))
    layers = []
    for r in range(n_rounds):
        for h in range(n_hubs):
            spoke = n_hubs + h * spokes_per_hub + (r % spokes_per_hub)
            layers.append(((h, spoke),))
    return qubits, layers


def _run_strategy(strategy, layout, stages, lookahead_max=12):
    state = ConcreteState(
        occupied=frozenset(),
        layout=tuple(layout),
        move_count=tuple(0 for _ in layout),
    )
    n_lanes = 0
    n_transitions = 0
    for i, stage in enumerate(stages):
        if not stage:
            continue
        c = tuple(c for c, _ in stage)
        t = tuple(t for _, t in stage)
        la = tuple(
            (tuple(c2 for c2, _ in s), tuple(t2 for _, t2 in s))
            for s in stages[i + 1 : i + 1 + lookahead_max]
            if s
        )
        new = strategy.cz_placements(state, c, t, la)
        if isinstance(new, ExecuteCZ) or hasattr(new, "move_layers"):
            n_lanes += sum(len(L) for L in new.move_layers)
            n_transitions += 1
            state = new
    return n_lanes, n_transitions


@pytest.mark.parametrize("H,sp,R", [(3, 6, 3), (3, 8, 3), (4, 6, 3), (4, 8, 3)])
def test_beats_default_on_hub_swap_chain(H, sp, R):
    """Lookahead-aware target picking beats Default on multi-hub patterns
    by 1.20-1.44× lane reduction at H>=3, sp>=6.

    Regression bound (loose): adapt-aware should not be worse than Default
    by more than 5%.
    """
    arch = get_physical_arch_spec()
    qubits, stages = _hub_swap_chain(H, sp, R)
    layout = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=arch
    ).compute_layout(qubits, stages)

    default_strat = PhysicalPlacementStrategy(
        arch_spec=arch,
        traversal=RustPlacementTraversal(strategy="astar", max_expansions=300),
        target_generator=DefaultTargetGenerator(),
    )
    la_strat = PhysicalPlacementStrategy(
        arch_spec=arch,
        traversal=RustPlacementTraversal(strategy="astar", max_expansions=300),
        target_generator=LookaheadCongestionAwareTargetGenerator(K=6, gamma=0.6),
    )

    default_lanes, default_trans = _run_strategy(default_strat, layout, stages)
    la_lanes, la_trans = _run_strategy(la_strat, layout, stages)

    assert (
        la_trans == default_trans
    ), f"transitions differ: default={default_trans}, la={la_trans}"
    # On the empirically dramatic configs the win is at least 1.10×.
    if (H, sp) == (4, 8):
        assert default_lanes / la_lanes >= 1.40
    elif (H, sp) == (3, 8):
        assert default_lanes / la_lanes >= 1.25
    elif (H, sp) == (3, 6):
        assert default_lanes / la_lanes >= 1.20
    else:  # (4, 6)
        assert default_lanes / la_lanes >= 1.15


# ---------------------------------------------------------------------- #
# 4. Behavioural — GHZ ladder (extra stages placed)                       #
# ---------------------------------------------------------------------- #


def _ghz_ladder(n):
    return tuple(range(n)), [((i, i + 1),) for i in range(n - 1)]


def test_places_more_stages_on_ghz_n_80():
    """On GHZ n=80, lookahead-aware places +3 more transitions than
    default (5.4% throughput improvement) — the empirical headline win.
    """
    arch = get_physical_arch_spec()
    qubits, stages = _ghz_ladder(80)
    layout = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=arch
    ).compute_layout(qubits, stages)

    default_strat = PhysicalPlacementStrategy(
        arch_spec=arch,
        traversal=RustPlacementTraversal(strategy="astar", max_expansions=300),
        target_generator=DefaultTargetGenerator(),
    )
    la_strat = PhysicalPlacementStrategy(
        arch_spec=arch,
        traversal=RustPlacementTraversal(strategy="astar", max_expansions=300),
        target_generator=LookaheadCongestionAwareTargetGenerator(K=3, gamma=0.7),
    )

    _, default_trans = _run_strategy(default_strat, layout, stages)
    _, la_trans = _run_strategy(la_strat, layout, stages)

    # Lookahead-aware places at least +1 more stage on GHZ n=80
    # (on the empirical baseline it places +3).
    assert la_trans > default_trans
