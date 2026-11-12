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
print("Lanes compiles independently of Gemini")
