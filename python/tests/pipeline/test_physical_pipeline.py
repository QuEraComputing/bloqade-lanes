"""Tests for the physical pipeline and NewPinnedQubit."""

import bloqade.squin as squin
import pytest
from kirin import ir, rewrite
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

from bloqade import qubit
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move, place
from bloqade.lanes.pipeline import PhysicalPipeline
from bloqade.lanes.rewrite.circuit2place import RewriteQubitsToPinnedQubits


def test_new_pinned_qubit_unpinned():
    """NewPinnedQubit with no address has location_address=None."""
    stmt = place.NewPinnedQubit()
    assert stmt.location_address is None


def test_new_pinned_qubit_pinned():
    """NewPinnedQubit with a LocationAddress stores it correctly."""
    addr = LocationAddress(word_id=4, site_id=2, zone_id=0)
    stmt = place.NewPinnedQubit(location_address=addr)
    assert stmt.location_address == addr


def test_rewrite_new_to_pinned():
    """qubit.stmts.New is lowered to NewPinnedQubit(location_address=None)."""
    block = ir.Block()
    plain_new = qubit.stmts.New()
    block.stmts.append(plain_new)

    rewrite.Walk(RewriteQubitsToPinnedQubits()).rewrite(block)

    pinned = [s for s in block.stmts if isinstance(s, place.NewPinnedQubit)]
    assert len(pinned) == 1
    assert pinned[0].location_address is None


def test_physical_pipeline_smoke():
    """Single-qubit kernel with terminal measure compiles end-to-end."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(1)
        squin.qubit.measure(ilist.IList([reg[0]]))  # type: ignore[arg-type]

    out = PhysicalPipeline().emit(kernel)
    assert out is not None
    fills = [s for s in out.callable_region.walk() if isinstance(s, move.Fill)]
    assert len(fills) == 1


def test_physical_pipeline_placement_strategy_default_is_none():
    """PhysicalPipeline.placement_strategy defaults to None; emit() constructs it from arch_spec."""
    pipeline = PhysicalPipeline()
    assert pipeline.placement_strategy is None


def test_physical_pipeline_no_new_pinned_remaining():
    """After compilation, no place.NewPinnedQubit statements remain."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.qubit.measure(ilist.IList([reg[0], reg[1]]))  # type: ignore[arg-type]

    out = PhysicalPipeline().emit(kernel)
    pinned_stmts = [
        s for s in out.callable_region.walk() if isinstance(s, place.NewPinnedQubit)
    ]
    assert pinned_stmts == []


def test_physical_pipeline_missing_measure_raises():
    """Kernel with no terminal measure raises at validation."""

    @squin.kernel
    def kernel():
        _reg = squin.qalloc(1)  # noqa: F841

    import pytest

    with pytest.raises(ValidationErrorGroup):
        PhysicalPipeline().emit(kernel, no_raise=False)


def test_physical_pipeline_no_raise_succeeds_without_terminal_measure():
    """With no_raise=True, PhysicalPipeline.emit must not raise even when there
    is no terminal measurement (validation errors are suppressed in lenient mode)."""

    @squin.kernel
    def kernel():
        squin.qalloc(1)  # no measurement

    # Should not raise in lenient mode.
    result = PhysicalPipeline().emit(kernel, no_raise=True)
    assert result is not None


def test_physical_pipeline_resolves_none_to_physical_defaults(monkeypatch):
    """When layout_heuristic and placement_strategy are None, PhysicalPipeline
    resolves them to PhysicalLayoutHeuristicGraphPartitionCenterOut wrapped in
    PalindromePlacementStrategy(NoHomePlacementStrategy) before passing them
    to the place→move stage."""
    from bloqade.lanes.analysis.placement import PalindromePlacementStrategy
    from bloqade.lanes.heuristics.physical.layout import (
        PhysicalLayoutHeuristicGraphPartitionCenterOut,
    )
    from bloqade.lanes.heuristics.physical.nohome import NoHomePlacementStrategy
    from bloqade.lanes.pipeline.base import _PlaceToMove

    captured: dict = {}
    _orig_emit = _PlaceToMove.emit

    def spy_emit(self_inner, mt, no_raise=True):
        captured["layout_heuristic_type"] = type(self_inner.layout_heuristic)
        captured["placement_strategy"] = self_inner.placement_strategy
        return _orig_emit(self_inner, mt, no_raise=no_raise)

    monkeypatch.setattr(_PlaceToMove, "emit", spy_emit)

    @squin.kernel
    def kernel():
        reg = squin.qalloc(1)
        squin.qubit.measure(ilist.IList([reg[0]]))  # type: ignore[arg-type]

    PhysicalPipeline().emit(kernel)

    assert (
        captured["layout_heuristic_type"]
        is PhysicalLayoutHeuristicGraphPartitionCenterOut
    )
    strategy = captured["placement_strategy"]
    assert isinstance(strategy, PalindromePlacementStrategy)
    assert isinstance(strategy.inner, NoHomePlacementStrategy)


def test_physical_pipeline_layout_heuristic_mismatch_warns():
    """resolved_layout_heuristic warns when the explicit heuristic carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
    from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic

    physical_arch = PhysicalPipeline().arch_spec
    logical_arch = get_logical_arch_spec()
    assert physical_arch != logical_arch

    mismatched_heuristic = LogicalLayoutHeuristic(arch_spec=logical_arch)
    pipeline = PhysicalPipeline(layout_heuristic=mismatched_heuristic)

    with pytest.warns(
        UserWarning, match="layout_heuristic was constructed with a different"
    ):
        result = pipeline.resolved_layout_heuristic

    assert result is mismatched_heuristic


def test_physical_pipeline_placement_strategy_mismatch_warns():
    """resolved_placement_strategy warns when the explicit strategy carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec
    from bloqade.lanes.heuristics.logical.placement import (
        LogicalPlacementStrategyNoHome,
    )

    physical_arch = PhysicalPipeline().arch_spec
    logical_arch = get_logical_arch_spec()
    assert physical_arch != logical_arch

    mismatched_strategy = LogicalPlacementStrategyNoHome(arch_spec=logical_arch)
    pipeline = PhysicalPipeline(placement_strategy=mismatched_strategy)

    with pytest.warns(
        UserWarning, match="placement_strategy was constructed with a different"
    ):
        result = pipeline.resolved_placement_strategy

    assert result is mismatched_strategy
