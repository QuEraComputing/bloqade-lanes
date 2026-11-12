import importlib.abc
import sys


class NoGemini(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bloqade.gemini" or fullname.startswith("bloqade.gemini."):
            raise AssertionError(fullname)


sys.meta_path.insert(0, NoGemini())

import pytest

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
from kirin.ir.exception import ValidationErrorGroup

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


# Valid programs alone cannot detect validation handlers silently falling back.
# Match the intended diagnostics so a different rejection cannot hide a missing
# table (e.g. "Gate t is not supported" instead of the state-preparation rule).
@squin.kernel.add(steane713)
def logical_late_t():
    q = squin.qalloc(1)
    squin.h(q[0])
    squin.t(q[0])
    return steane713.terminal_measure(q)


@squin.kernel.add(arch)
def physical_duplicate():
    a = arch.new_at(0, 0, 0)
    b = arch.new_at(0, 0, 0)
    return squin.broadcast.measure(ilist.IList([a, b]))


for source, pipeline, diagnostic in (
    (logical_late_t, LogicalPipeline(), "Non-clifford gate t"),
    (physical_duplicate, PhysicalPipeline(), "pinned by two"),
):
    AggressiveUnroll(source.dialects).fixpoint(source)
    with pytest.raises(ValidationErrorGroup, match=diagnostic):
        pipeline.emit(source, no_raise=False)

assert not any(n.startswith("bloqade.gemini") for n in sys.modules)

print("Lanes compiles and validates independently of Gemini")
