from kirin import ir, types
from kirin.dialects import py, scf
from tests._qmove_helpers import blocks_equal

ITERABLE = ir.TestValue(types.Any)


def _loop(body_value: int) -> ir.Block:
    body = ir.Block()
    body.args.append_from(types.Int, "i")
    body.stmts.append(py.Constant(body_value))
    body.stmts.append(scf.Yield())
    return ir.Block([scf.For(ITERABLE, ir.Region(body))])


def test_identical_ir_matches():
    assert blocks_equal(_loop(1), _loop(1)) is None


def test_difference_inside_a_nested_region_is_caught():
    a, b = _loop(1), _loop(2)
    # kirin's own comparison misses this: it never looks inside nested regions.
    assert a.is_structurally_equal(b)
    assert blocks_equal(a, b) is not None


def test_result_type_difference_is_caught():
    float_typed = py.Constant(1)
    float_typed.result.type = types.Float
    assert blocks_equal(ir.Block([py.Constant(1)]), ir.Block([float_typed])) is not None
