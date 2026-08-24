"""Tests for the shared TransformABC.emit dialect-group guard."""

from dataclasses import dataclass

import pytest
from kirin import ir, types
from kirin.dialects import func

from bloqade.lanes.bytecode.encoding import ZoneAddress
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.transform.base import TransformABC


def _method_with_move_stmt(dialects: list) -> ir.Method:
    cz = move.ConstZone(value=ZoneAddress(0))
    ret = func.Return(cz.result)
    block = ir.Block(argtypes=(types.MethodType,))
    for s in (cz, ret):
        block.stmts.append(s)
    function = func.Function(
        sym_name="main",
        signature=func.Signature((), types.Any),
        slots=(),
        body=ir.Region(blocks=block),
    )
    return ir.Method(
        dialects=ir.DialectGroup(dialects),
        code=function,
        sym_name="main",
        arg_names=[],
    )


@dataclass
class DiscardMove(TransformABC):
    """A stage that drops ``move`` from the group without lowering anything."""

    def _emit(self, mt: ir.Method, no_raise: bool = True) -> ir.Method:
        return mt.similar(mt.dialects.discard(move.dialect))


def test_emit_raises_on_leftover_statements():
    method = _method_with_move_stmt([stack_move.dialect, move.dialect, func.dialect])

    with pytest.raises(ValueError, match="DiscardMove.*ConstZone"):
        DiscardMove().emit(method, no_raise=False)


def test_emit_is_silent_when_no_raise():
    """no_raise=True is the best-effort mode: incomplete lowering is tolerated."""
    method = _method_with_move_stmt([stack_move.dialect, move.dialect, func.dialect])

    out = DiscardMove().emit(method)

    assert move.dialect not in out.dialects


def test_emit_returns_result_when_clean():
    cf = stack_move.ConstFloat(value=0.5)
    ret = func.Return(cf.result)
    block = ir.Block(argtypes=(types.MethodType,))
    for s in (cf, ret):
        block.stmts.append(s)
    function = func.Function(
        sym_name="main",
        signature=func.Signature((), types.Any),
        slots=(),
        body=ir.Region(blocks=block),
    )
    method = ir.Method(
        dialects=ir.DialectGroup([stack_move.dialect, move.dialect, func.dialect]),
        code=function,
        sym_name="main",
        arg_names=[],
    )

    out = DiscardMove().emit(method, no_raise=False)

    assert move.dialect not in out.dialects
