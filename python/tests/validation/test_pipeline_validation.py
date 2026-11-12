"""Pipeline-level tests for Task E3: eager compiler validation in NativeToPlace.emit.

These tests verify that both the per-statement (E1) and cross-statement
duplicate-address (E2) validators run eagerly via LogicalPipeline before the
circuit→place lowering, and that errors surface as ValidationErrorGroup.
"""

import bloqade.squin as squin
import pytest
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

from bloqade.lanes.dialects import steane713
from bloqade.lanes.dialects.arch import new_at
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.transform import LogicalPipeline

from ._steane_kernel import kernel as steane_kernel

# ---------------------------------------------------------------------------
# Common heuristic / strategy fixtures
# ---------------------------------------------------------------------------

_LAYOUT = LogicalLayoutHeuristic()
_PLACEMENT = LogicalPlacementStrategyNoHome()


def _compile(kernel):
    """Run LogicalPipeline with the standard logical heuristic pair."""
    return LogicalPipeline(
        layout_heuristic=_LAYOUT,
        placement_strategy=_PLACEMENT,
    ).emit(kernel, no_raise=False)


# ---------------------------------------------------------------------------
# 1. Pipeline catches const-prop failure
# ---------------------------------------------------------------------------


def test_pipeline_catches_non_const_arg():
    """A new_at whose zone_id is a function argument (not compile-time const)
    should cause LogicalPipeline to raise with a 'compile-time constant' message.
    """

    @steane_kernel
    def kernel(z: int):
        q = new_at(z, 0, 0)  # z is a function arg — not const
        return steane713.terminal_measure(ilist.IList([q]))

    with pytest.raises(ValidationErrorGroup) as exc_info:
        _compile(kernel)

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("compile-time constant" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# 2. Pipeline catches range failure
# ---------------------------------------------------------------------------


def test_pipeline_catches_out_of_range_address():
    """new_at(99, 0, 0) with zone 99 outside the arch should raise with an
    'Invalid location address' message.
    """

    @steane_kernel
    def kernel():
        q = new_at(99, 0, 0)  # zone 99 is out of range
        return steane713.terminal_measure(ilist.IList([q]))

    with pytest.raises(ValidationErrorGroup) as exc_info:
        _compile(kernel)

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("Invalid location address" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# 3. Pipeline catches duplicate addresses
# ---------------------------------------------------------------------------


def test_pipeline_catches_duplicate_addresses():
    """Two new_at calls pinning the same address should raise with 'pinned by two'."""

    @steane_kernel
    def kernel():
        q0 = new_at(0, 0, 0)
        q1 = new_at(0, 0, 0)  # same address — duplicate
        return steane713.terminal_measure(ilist.IList([q0, q1]))

    with pytest.raises(ValidationErrorGroup) as exc_info:
        _compile(kernel)

    errors = exc_info.value.errors
    assert len(errors) >= 1
    assert any("pinned by two" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# 4. Valid kernel compiles end-to-end without errors
# ---------------------------------------------------------------------------


def test_pipeline_valid_kernel_compiles():
    """A Steane kernel with no new_at calls should compile without validation errors."""

    @steane_kernel
    def kernel():
        q = squin.qalloc(2)
        squin.h(q[0])
        squin.cx(q[0], q[1])
        steane713.terminal_measure(q)

    # Should not raise.
    _compile(kernel)
