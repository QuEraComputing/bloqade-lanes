import importlib.abc
import sys


class NoGemini(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bloqade.gemini" or fullname.startswith("bloqade.gemini."):
            raise AssertionError(fullname)


sys.meta_path.insert(0, NoGemini())

from bloqade import squin
from bloqade.lanes.dialects import arch
from bloqade.lanes.transform.native_to_place import NativeToPlace


@squin.kernel
def generic():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])


NativeToPlace().emit(generic, no_raise=False)


@squin.kernel.add(arch)
def location_only():
    q = squin.qalloc(1)
    squin.h(q[0])
    return arch.loc(0, 0, 0)


NativeToPlace().emit(location_only, no_raise=False)
assert not any(n.startswith("bloqade.gemini") for n in sys.modules)

from bloqade.rewrite.passes import AggressiveUnroll

# Exercise the actual pipelines, including their validation and analysis
# dispatch. NativeToPlace alone did not cover the missing Steane tables.
from kirin.dialects import ilist

from bloqade.lanes.dialects import move, steane713
from bloqade.lanes.transform import LogicalPipeline, PhysicalPipeline


@squin.kernel.add(steane713)
def logical():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return steane713.terminal_measure(q)


@squin.kernel.add(arch).add(steane713)
def pinned_logical():
    q = ilist.IList([arch.new_at(0, 4, 0)])
    squin.h(q[0])
    return steane713.terminal_measure(q)


@squin.kernel.add(arch)
def physical():
    q = ilist.IList([arch.new_at(0, 0, 0), arch.new_at(0, 0, 1)])
    squin.h(q[0])
    squin.cz(q[0], q[1])
    return squin.broadcast.measure(q)


for source, pipeline in (
    (logical, LogicalPipeline()),
    (pinned_logical, LogicalPipeline()),
    (physical, PhysicalPipeline()),
):
    AggressiveUnroll(source.dialects).fixpoint(source)
    compiled = pipeline.emit(source, no_raise=False)
    compiled.verify()
    assert any(stmt.dialect is move.dialect for stmt in compiled.code.walk())

assert not any(n.startswith("bloqade.gemini") for n in sys.modules)

print("Lanes compiles independently of Gemini")
