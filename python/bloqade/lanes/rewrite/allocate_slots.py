"""Expand compiler-configured code-block slots into pinned physical allocations."""

from dataclasses import dataclass

from kirin import ir
from kirin.analysis import const
from kirin.dialects import ilist, py
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects.arch import Loc, SiteId, WordId, ZoneId
from bloqade.lanes.dialects.arch.stmts import NewAt
from bloqade.lanes.dialects.code_block import QallocAt, Register
from bloqade.lanes.validation.recursion import CallGraph


@dataclass
class AllocateSlots(RewriteRule):
    slots: list[list[tuple[int, int, int]]] | None
    arch_spec: ArchSpec | None

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, QallocAt):
            return RewriteResult()
        if self.slots is None or self.arch_spec is None:
            raise ValueError("qalloc_at requires compiler slots and arch_spec")
        hint = node.slots.hints.get("const")
        if not isinstance(hint, const.Value) or not isinstance(hint.data, ilist.IList):
            raise TypeError(
                "qalloc_at slot indices must be a compile-time constant IList"
            )

        blocks = []
        for index in hint.data:
            if type(index) is not int or not 0 <= index < len(self.slots):
                raise ValueError(f"qalloc_at slot index {index!r} is out of range")
            coordinates = self.slots[index]
            if not coordinates:
                raise ValueError(f"qalloc_at slot {index} is empty")
            qubits = []
            for coordinate in coordinates:
                if (
                    not isinstance(coordinate, (tuple, list))
                    or len(coordinate) != 3
                    or any(type(value) is not int for value in coordinate)
                ):
                    raise ValueError(
                        f"qalloc_at slot {index}: expected (zone, row, col), got {coordinate!r}"
                    )
                if self.arch_spec.location_at(*coordinate) is None:
                    raise ValueError(
                        f"qalloc_at slot {index}: no architecture location at {coordinate!r}"
                    )
                coords = [py.Constant(value) for value in coordinate]
                for stmt in coords:
                    stmt.insert_before(node)
                location = Loc(*(stmt.result for stmt in coords))
                location.arch_spec = self.arch_spec
                location.insert_before(node)
                address = [
                    ZoneId(location.result),
                    WordId(location.result),
                    SiteId(location.result),
                ]
                for stmt in address:
                    stmt.insert_before(node)
                allocation = NewAt(*(stmt.result for stmt in address))
                allocation.insert_before(node)
                qubits.append(allocation.qubit)
            block = ilist.New(tuple(qubits))
            block.insert_before(node)
            Register(block.result).insert_before(node)
            blocks.append(block.result)
        node.replace_by(ilist.New(tuple(blocks)))
        return RewriteResult(has_done_something=True)


def has_slot_allocations(method: ir.Method) -> bool:
    """Include helper kernels that the frontend has not inlined yet."""
    graph = CallGraph(method)
    methods = {method, *graph.edges}
    for callees in graph.edges.values():
        methods.update(callees)
    return any(
        isinstance(stmt, QallocAt)
        for callee in methods
        for stmt in callee.callable_region.walk()
    )
