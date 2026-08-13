"""Reject Gemini kernels whose call graph contains a cycle.

A Gemini kernel is lowered to a fixed sequence of physical atom moves, so a
recursive call graph has no finite lowering and cannot be compiled. Rejecting it
is therefore not a limitation we are working around but the correct answer.

Failing *early* matters for a second reason: every pass downstream diverges on a
cyclic call graph rather than erroring. ``AddressAnalysis`` re-analyses each
callee at every call site with no memoisation, so a branching cycle costs
``phi ** max_depth`` interpreter calls -- and because kirin's depth guard returns
bottom instead of raising, the compile presents as an unkillable hang with no
diagnostic. See bloqade-lanes#921 and bloqade-circuit#852.

``CallGraph`` below is deliberately self-contained rather than reusing
``kirin.analysis.CallGraph``, which recurses without a visited set and raises
``RecursionError`` on precisely the inputs this module exists to diagnose
(kirin#703). It also differs in two ways worth keeping if this is upstreamed:
edges are keyed caller -> callee (matching how they read), and callees are found
through the ``ir.StaticCall`` trait rather than an ``isinstance`` check against
``func.Invoke``, so any dialect that declares the trait participates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from kirin import ir
from kirin.dialects import func
from kirin.validation import ValidationPass


def _name(method: ir.Method) -> str:
    return method.sym_name or "<lambda>"


def _self_value(method: ir.Method) -> ir.SSAValue | None:
    """The ``%<name>_self`` block argument of ``method``'s callable region.

    A kernel that calls itself is lowered as a *dynamic* call on this value, and
    is only rewritten into a static ``func.Invoke`` once the symbol it refers to
    exists -- which happens after ``run_pass`` returns. A guard that runs before
    the other passes therefore has to recognise this form too.
    """
    trait = method.code.get_trait(ir.CallableStmtInterface)
    if trait is None:
        return None

    region = trait.get_callable_region(method.code)
    if not region.blocks or not region.blocks[0].args:
        return None

    return region.blocks[0].args[0]


def _parameter_values(method: ir.Method) -> dict[int, ir.SSAValue]:
    """``method``'s own parameters, keyed by id, excluding the self argument.

    A call through one of these has no statically knowable target, so it is what
    makes a dynamic call graph unanalysable. The self argument is excluded
    because it *is* known -- it is reported as recursion instead.
    """
    trait = method.code.get_trait(ir.CallableStmtInterface)
    if trait is None:
        return {}

    region = trait.get_callable_region(method.code)
    if not region.blocks:
        return {}

    return {id(arg): arg for arg in region.blocks[0].args[1:]}


def _sort_key(method: ir.Method) -> tuple[str, int]:
    # NOTE: id() only breaks ties between same-named methods; it keeps the
    # ordering total so diagnostics are stable within a single process.
    return _name(method), id(method)


def format_cycle(cycle: list[ir.Method]) -> str:
    """Render a cycle as ``a -> b -> a``, closing the loop for readability."""
    return " -> ".join([_name(method) for method in cycle] + [_name(cycle[0])])


@dataclass(init=False)
class CallGraph:
    """Static call graph rooted at ``entry``, keyed caller -> callees.

    Construction is an iterative worklist, so a recursive kernel produces a
    graph containing a cycle instead of exhausting the Python stack.

    Only statically resolved edges are followed. A dynamic ``func.Call`` through
    an SSA value is invisible here: resolving one needs the same constant
    propagation that diverges on cyclic input, so it cannot be used by a guard
    whose job is to run first.
    """

    entry: ir.Method
    edges: dict[ir.Method, set[ir.Method]] = field(default_factory=dict)

    def __init__(self, mt: ir.Method) -> None:
        self.entry = mt
        self.edges = {}

        worklist = [mt]
        while worklist:
            caller = worklist.pop()
            if caller in self.edges:
                continue

            callees: set[ir.Method] = set()
            self_value = _self_value(caller)
            for stmt in caller.code.walk():
                if trait := stmt.get_trait(ir.StaticCall):
                    callees.add(trait.get_callee(stmt))
                elif (
                    self_value is not None
                    and isinstance(stmt, func.Call)
                    and stmt.callee is self_value
                ):
                    # Not-yet-resolved self call; record it as a real self edge.
                    callees.add(caller)

            self.edges[caller] = callees
            worklist.extend(callee for callee in callees if callee not in self.edges)

    def callees(self, method: ir.Method) -> list[ir.Method]:
        """Callees of ``method``, in a deterministic order."""
        return sorted(self.edges.get(method, ()), key=_sort_key)

    def find_cycles(self) -> list[list[ir.Method]]:
        """Return one representative cycle per distinct set of mutual callers.

        Each cycle is ``[m0, ..., mn]`` where ``mn`` calls ``m0``; an empty list
        means the call graph is acyclic. The traversal is an explicit-stack DFS
        so that deeply nested (but acyclic) call graphs stay safe too.
        """
        cycles: list[list[ir.Method]] = []
        reported: set[frozenset[ir.Method]] = set()
        visited: set[ir.Method] = set()

        for root in sorted(self.edges, key=_sort_key):
            if root in visited:
                continue

            visited.add(root)
            path = [root]
            on_path = {root}
            stack: list[tuple[ir.Method, Iterator[ir.Method]]] = [
                (root, iter(self.callees(root)))
            ]

            while stack:
                _, pending = stack[-1]
                descended = False

                for callee in pending:
                    if callee in on_path:
                        # Back edge: everything from `callee` onwards is a cycle.
                        cycle = path[path.index(callee) :]
                        if (key := frozenset(cycle)) not in reported:
                            reported.add(key)
                            cycles.append(cycle)
                        continue

                    if callee in visited:
                        continue

                    visited.add(callee)
                    path.append(callee)
                    on_path.add(callee)
                    stack.append((callee, iter(self.callees(callee))))
                    descended = True
                    break

                if not descended:
                    stack.pop()
                    on_path.discard(path.pop())

        return cycles


@dataclass
class NoRecursionValidation(ValidationPass):
    """Report every recursive cycle reachable from the validated kernel."""

    def name(self) -> str:
        return "gemini.no_recursion"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        graph = CallGraph(method)
        # NOTE: blame the validated kernel rather than the statement that closes
        # the cycle. `ValidationSuite` renders hints against the method it was
        # given, so pointing at a statement inside a *callee* resolves to the
        # wrong source line.
        errors = [
            ir.ValidationError(
                method.code,
                "recursion is not supported in Gemini kernels; "
                f"call graph contains the cycle {format_cycle(cycle)}",
            )
            for cycle in graph.find_cycles()
        ]
        return graph, errors


@dataclass
class NoOpaqueCallValidation(ValidationPass):
    """Reject calls made through one of the kernel's own parameters.

    ``CallGraph`` can only see statically resolved edges, so a call through a
    parameter is a hole in it: passing a kernel in as a value and calling it
    builds a cycle that no amount of graph walking will find, and a cycle with
    branching factor >= 2 still costs ``phi ** max_depth`` in the address
    analysis. Resolving the target instead would need the constant propagation
    that diverges on exactly that input, so the shape is rejected rather than
    resolved.

    This costs nothing real: a Gemini kernel cannot lower a call through a value
    anyway, and no kernel in the test suite performs one.
    """

    def name(self) -> str:
        return "gemini.no_opaque_call"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        parameters = _parameter_values(method)
        errors: list[ir.ValidationError] = []

        for stmt in method.code.walk():
            if not isinstance(stmt, func.Call):
                continue

            parameter = parameters.get(id(stmt.callee))
            if parameter is None:
                continue

            name = parameter.name or "<unnamed>"
            errors.append(
                ir.ValidationError(
                    method.code,
                    f"calling the parameter '{name}' is not supported in Gemini "
                    "kernels; the call target must be known at compile time",
                )
            )

        return None, errors
