"""Tests for bloqade.lanes.utils helpers."""

from kirin import ir, types
from kirin.dialects import func

from bloqade.lanes.bytecode.encoding import ZoneAddress
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.utils import statements_outside_dialect_group


def _method(*stmts, dialects: list) -> ir.Method:
    block = ir.Block(argtypes=(types.MethodType,))
    for s in stmts:
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


def test_no_offenders_when_all_in_group():
    cf = stack_move.ConstFloat(value=0.5)
    ret = func.Return(cf.result)
    method = _method(cf, ret, dialects=[stack_move.dialect, func.dialect])

    assert statements_outside_dialect_group(method) == []


def test_detects_statement_outside_group():
    # move.ConstZone is present but move.dialect is NOT in the group.
    cz = move.ConstZone(value=ZoneAddress(0))
    ret = func.Return(cz.result)
    method = _method(cz, ret, dialects=[stack_move.dialect, func.dialect])

    offenders = statements_outside_dialect_group(method)

    assert offenders == [cz]


def test_no_offenders_when_dialect_present():
    # Same statement, but now move.dialect IS in the group → no offenders.
    cz = move.ConstZone(value=ZoneAddress(0))
    ret = func.Return(cz.result)
    method = _method(cz, ret, dialects=[stack_move.dialect, move.dialect, func.dialect])

    assert statements_outside_dialect_group(method) == []
