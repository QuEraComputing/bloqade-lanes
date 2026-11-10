import importlib.abc
import sys


class NoGemini(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bloqade.gemini" or fullname.startswith("bloqade.gemini."):
            raise AssertionError(fullname)


sys.meta_path.insert(0, NoGemini())

from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.lanes.dialects import arch, steane713
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.transform import LogicalPipeline, PhysicalPipeline

kernel = squin.kernel.add(arch).add(steane713).add(extensions)


@kernel
def logical():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return steane713.terminal_measure(q)


try:
    LogicalPipeline().emit(logical, no_raise=False)
except (ValueError, ValidationErrorGroup) as exc:
    assert "Missing compiler method tables" in str(exc), str(exc)
else:
    raise AssertionError("Logical compilation silently accepted missing tables")


@squin.kernel
def physical():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return squin.broadcast.measure(q)


from bloqade.lanes.transform.native_to_place import NativeToPlace


@squin.kernel
def generic():
    q = squin.qalloc(2)
    squin.h(q[0])
    squin.cx(q[0], q[1])


NativeToPlace().emit(generic, no_raise=False)
try:
    PhysicalPipeline().emit(physical, no_raise=False)
except (ValueError, ValidationErrorGroup) as exc:
    assert "Missing compiler method tables" in str(exc), str(exc)
else:
    raise AssertionError("Physical validation silently accepted missing tables")
assert not any(n.startswith("bloqade.gemini") for n in sys.modules)
print("Generic lowering works; specialized pipelines require frontend tables")
