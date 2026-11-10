"""Physical allocation of code blocks at compiler-configured slots."""

from typing import Any, TypeVar

from bloqade.types import Qubit, QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

N = TypeVar("N")
dialect = ir.Dialect("gemini.physical.qubit")


@statement(dialect=dialect)
class QallocAt(ir.Statement):
    """Allocate one physical code block for each selected slot, in list order."""

    traits = frozenset({lowering.FromPythonCall()})
    slots: ir.SSAValue = info.argument(ilist.IListType[types.Int, types.Any])
    result: ir.ResultValue = info.result(
        ilist.IListType[ilist.IListType[QubitType, types.Any], types.Any]
    )


@lowering.wraps(QallocAt)
def qalloc_at(
    slots: ilist.IList[int, N],
) -> ilist.IList[ilist.IList[Qubit, Any], N]:
    """Allocate physical qubits at zero-based indices into ``PhysicalPipeline.slots``.

    The input must be compile-time constant. Each returned inner IList contains
    the physical qubits of one selected code block, in the configured address
    order. Slot coordinates are supplied to the compiler, not to this kernel.
    """
    ...  # noqa: PIE790


def has_slot_allocations(method: ir.Method) -> bool:
    """Include helper kernels that the frontend has not inlined yet."""
    from bloqade.gemini.common.validation.recursion import CallGraph

    graph = CallGraph(method)
    methods = {method, *graph.edges}
    for callees in graph.edges.values():
        methods.update(callees)
    return any(
        isinstance(stmt, QallocAt)
        for callee in methods
        for stmt in callee.callable_region.walk()
    )
