from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from kirin import ir
from kirin.dialects import py

from bloqade import squin
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import arch
from bloqade.lanes.dialects.arch.rewrite import RewriteLocationAttr


def test_registered_inference_rule_supports_concurrent_compilation(monkeypatch):
    kernel = squin.kernel.add(arch)

    @kernel(typeinfer=False, fold=False)
    def first(location: LocationAddress):
        return location.word_id

    @kernel(typeinfer=False, fold=False)
    def second(location: LocationAddress):
        return location.site_id

    barrier = Barrier(2)
    original = RewriteLocationAttr.rewrite_Statement

    def synchronized_rewrite(self, node: ir.Statement):
        if isinstance(node, py.GetAttr):
            barrier.wait(timeout=5)
        return original(self, node)

    monkeypatch.setattr(RewriteLocationAttr, "rewrite_Statement", synchronized_rewrite)
    registered_rule = arch.dialect.rules.inference[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(registered_rule.rewrite, [first.code, second.code]))

    assert all(result.has_done_something for result in results)
    for method, expected in [(first, arch.WordId), (second, arch.SiteId)]:
        statements = list(method.code.walk())
        assert any(isinstance(stmt, expected) for stmt in statements)
        assert not any(isinstance(stmt, py.GetAttr) for stmt in statements)
