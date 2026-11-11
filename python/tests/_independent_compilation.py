import importlib.abc
import sys


class NoGemini(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bloqade.gemini" or fullname.startswith("bloqade.gemini."):
            raise AssertionError(fullname)


sys.meta_path.insert(0, NoGemini())

from bloqade.rewrite.passes import AggressiveUnroll
from kirin import rewrite
from kirin.dialects import ilist

from bloqade import squin
from bloqade.lanes.dialects import arch, steane713
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.rewrite.qubit_count import InsertQubitCount
from bloqade.lanes.transform import LogicalPipeline, PhysicalPipeline

kernel = squin.kernel.add(arch).add(steane713).add(extensions)


@kernel
def logical():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return steane713.terminal_measure(q)


# Compiler method tables are registered by lanes, so specialized pipelines
# compile without a frontend having been imported first. Prepare the logical
# kernel the way a frontend decorator would: flatten it and record the Steane
# [[7,1,3]] physical width on its terminal measurement.
AggressiveUnroll(logical.dialects, no_raise=False).fixpoint(logical)
rewrite.Walk(InsertQubitCount(7)).rewrite(logical.code)
LogicalPipeline().emit(logical, no_raise=False)


@squin.kernel
def physical():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return squin.broadcast.measure(q)


PhysicalPipeline().emit(physical, no_raise=False)

pinned = squin.kernel.add(arch)


@pinned
def physical_pinned():
    a = arch.new_at(0, 0, 0)
    b = arch.new_at(0, 0, 1)
    squin.h(a)
    squin.cx(a, b)
    return squin.broadcast.measure(ilist.IList([a, b]))


PhysicalPipeline().emit(physical_pinned, no_raise=False)


from bloqade.lanes.transform.native_to_place import NativeToPlace


@squin.kernel
def generic():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])


NativeToPlace().emit(generic, no_raise=False)


@pinned
def location_only():
    q = squin.qalloc(1)
    squin.h(q[0])
    return arch.loc(0, 0, 0)


NativeToPlace().emit(location_only, no_raise=False)
assert not any(n.startswith("bloqade.gemini") for n in sys.modules)
print("Lanes compiles independently of Gemini")
