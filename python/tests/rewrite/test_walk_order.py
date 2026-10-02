"""Pin the visit order of ``kirin.rewrite.Walk``.

``FlatBlockValidation`` (``bloqade.lanes.validation.flat_block``) exists because
``Walk`` hands a rule statements in execution order only within one block. These
tests pin the two places it does not, so a kirin upgrade that changes either one
fails here rather than silently changing what that validation guards against.

The behaviour is kirin's ``Walk.populate_worklist_Region`` /
``populate_worklist_Statement`` (``kirin/rewrite/walk.py``). The worklist is a
FIFO queue, so the order nodes are enqueued in is the order the rule sees them.
Both tests use the defaults, ``reverse=False`` and ``region_first=False``.
"""

from dataclasses import dataclass, field

from kirin import ir, types as kirin_types
from kirin.dialects import func, py
from kirin.rewrite import Walk
from kirin.rewrite.abc import RewriteResult, RewriteRule


@dataclass
class Record(RewriteRule):
    """Record the name of every node ``Walk`` visits."""

    names: dict[int, str]
    visited: list[str] = field(default_factory=list)

    def rewrite(self, node: ir.IRNode) -> RewriteResult:
        self.visited.append(self.names[id(node)])
        return RewriteResult()


def _recorder(**nodes: ir.IRNode) -> Record:
    return Record({id(node): name for name, node in nodes.items()})


def test_a_regions_blocks_are_visited_last_to_first():
    """A rule carrying state forward sees the later block's statements first."""
    c1, c2, ret = py.Constant(1), py.Constant(2), func.Return()
    first, second = ir.Block([c1]), ir.Block([c2, ret])
    region = ir.Region([first, second])

    rule = _recorder(region=region, first=first, c1=c1, second=second, c2=c2, ret=ret)
    Walk(rule).rewrite(region)

    assert rule.visited == ["region", "second", "c2", "ret", "first", "c1"]


def test_a_statements_regions_are_visited_before_the_statement():
    """A nested function's region is reached mid-block, before the
    ``func.Function`` owning it, so per-region state resets between ``a`` and
    ``b``."""
    x, inner_ret = py.Constant(10), func.Return()
    inner_block = ir.Block([x, inner_ret])
    inner_region = ir.Region(inner_block)
    inner = func.Function(
        sym_name="inner",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=inner_region,
    )
    a, b = py.Constant(1), py.Constant(2)
    block = ir.Block([a, inner, b])

    rule = _recorder(
        block=block,
        a=a,
        inner_region=inner_region,
        inner_block=inner_block,
        x=x,
        inner_ret=inner_ret,
        inner=inner,
        b=b,
    )
    Walk(rule).rewrite(block)

    assert rule.visited == [
        "block",
        "a",
        "inner_region",
        "inner_block",
        "x",
        "inner_ret",
        "inner",
        "b",
    ]
