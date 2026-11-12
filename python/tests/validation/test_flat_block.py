"""Tests for FlatBlockValidation."""

from kirin import ir, types as kirin_types
from kirin.dialects import func, py
from kirin.validation import ValidationSuite

from bloqade import squin
from bloqade.lanes.validation.flat_block import FlatBlockValidation


def _method(*blocks: ir.Block) -> ir.Method:
    """A minimal method whose callable region holds ``blocks``."""

    @squin.kernel
    def stub():
        return None

    out = stub.similar()
    out.code = func.Function(
        sym_name="stub",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(list(blocks)),
    )
    return out


def _errors(method: ir.Method) -> list[str]:
    result = ValidationSuite([FlatBlockValidation]).validate(method)
    return [] if result.is_valid else ["invalid"] * result.error_count()


def test_a_single_block_region_is_valid():
    assert _errors(_method(ir.Block([py.Constant(1), func.Return()]))) == []


def test_a_multi_block_region_is_rejected():
    """Walk visits blocks in reverse, so a frame spanning them accumulates
    backwards -- see test_walk_order.py."""
    assert (
        _errors(
            _method(
                ir.Block([py.Constant(1)]), ir.Block([py.Constant(2), func.Return()])
            )
        )
        != []
    )


def test_a_nested_function_is_rejected():
    """Walk reaches a nested function's region before the statement owning it,
    so a rule resetting per-region state has it wiped mid-block."""
    inner = func.Function(
        sym_name="inner",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(ir.Block([func.ConstantNone(), func.Return()])),
    )
    assert _errors(_method(ir.Block([inner, func.Return()]))) != []


def test_a_multi_block_region_nested_in_a_statement_is_rejected():
    """The scan covers every region, not just the callable one."""
    inner = func.Function(
        sym_name="inner",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region([ir.Block([py.Constant(1)]), ir.Block([func.Return()])]),
    )
    # Two errors: the nested function, and its two-block region.
    assert len(_errors(_method(ir.Block([inner, func.Return()])))) == 2
