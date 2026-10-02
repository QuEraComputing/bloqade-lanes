from kirin import ir, rewrite, types
from kirin.dialects import func, py

from bloqade import squin
from bloqade.lanes.rewrite.const_call_to_invoke import ConstCallToInvoke


@squin.kernel
def _callee(x: int) -> int:
    return x + 1


def test_call_of_a_constant_method_becomes_invoke():
    arg = ir.TestValue(types.Int)
    const = py.Constant(_callee)
    call = func.Call(const.result, (arg,), kwargs=())
    block = ir.Block([const, call])

    assert rewrite.Walk(ConstCallToInvoke()).rewrite(block).has_done_something
    (invoke,) = [s for s in block.stmts if isinstance(s, func.Invoke)]
    assert invoke.callee is _callee
    assert tuple(invoke.inputs) == (arg,)


def test_call_of_a_non_constant_callee_is_left_alone():
    callee = ir.TestValue(types.Any)
    call = func.Call(callee, (ir.TestValue(types.Int),), kwargs=())
    block = ir.Block([call])
    assert not rewrite.Walk(ConstCallToInvoke()).rewrite(block).has_done_something


def test_rewrites_inside_loop_bodies():
    from kirin.dialects import scf

    const = py.Constant(_callee)
    body = ir.Block()
    i = body.args.append_from(types.Int, "i")
    call = func.Call(const.result, (i,), kwargs=())
    body.stmts.append(call)
    body.stmts.append(scf.Yield())
    loop = scf.For(ir.TestValue(types.Any), ir.Region(body))
    block = ir.Block([const, loop])

    assert rewrite.Walk(ConstCallToInvoke()).rewrite(block).has_done_something
    assert any(isinstance(s, func.Invoke) for s in body.stmts)
