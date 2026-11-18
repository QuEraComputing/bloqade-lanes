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

from kirin import ir, types
from kirin.dialects import func
from kirin.validation import ValidationPass, ValidationSuite


def _callable_operands(stmt: ir.Statement) -> list[ir.SSAValue]:
    """Arguments of ``stmt`` that hold a callable.

    Higher-order statements such as ``ilist.Map``, ``Foldl``, ``Foldr``,
    ``Scan`` and ``ForEach`` invoke their ``fn`` argument, but carry no
    ``ir.StaticCall`` trait to say so -- kirin does not mark them as calls. A
    trait-driven walk therefore misses every edge through them, in both
    directions: a recursion routed through ``ilist.map`` is invisible, and the
    call graph is incomplete even for correct code.

    Keying on the operand *type* instead of a trait catches all of them, and any
    future dialect that takes a callable, without needing a list of statements.

    Bottom has to be excluded explicitly: it is a subtype of everything, so an
    operand whose type inference gave up matches ``MethodType`` and would be
    reported as an opaque call. Real kernels hit this -- `set_detector`'s
    `coordinates` argument is one.
    """
    return [
        arg
        for arg in stmt.args
        if arg.type.is_subseteq(types.MethodType)
        and not arg.type.is_subseteq(types.Bottom)
    ]


def _constant_method(value: ir.SSAValue) -> ir.Method | None:
    """The method ``value`` holds, when it is a compile-time constant."""
    attribute = getattr(value.owner, "value", None)
    if attribute is None:
        return None

    data = attribute.unwrap() if hasattr(attribute, "unwrap") else attribute
    return data if isinstance(data, ir.Method) else None


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


def _parameter_values(method: ir.Method) -> set[ir.SSAValue]:
    """``method``'s own parameters, excluding the self argument.

    A call through one of these has no statically knowable target, so it is what
    makes a dynamic call graph unanalysable. The self argument is excluded
    because it *is* known -- it is reported as recursion instead.

    Membership is identity in practice: ``SSAValue.__hash__`` is ``id``, so two
    distinct arguments never share a bucket even where the dataclass-generated
    ``__eq__`` would compare their fields.
    """
    trait = method.code.get_trait(ir.CallableStmtInterface)
    if trait is None:
        return set()

    region = trait.get_callable_region(method.code)
    if not region.blocks:
        return set()

    return set(region.blocks[0].args[1:])


def _sort_key(method: ir.Method) -> tuple[str, int]:
    # NOTE: id() only breaks ties between same-named methods; it keeps the
    # ordering total so diagnostics are stable within a single process.
    return _name(method), id(method)


@dataclass(frozen=True)
class Cycle:
    """A cycle, plus how the validated kernel reaches it.

    ``members`` is ``[m0, ..., mn]`` where ``mn`` calls ``m0``. ``route`` is the
    path from the entry method down to ``m0``, excluding the cycle itself, and
    is empty when the entry method is part of the cycle.

    The route matters because the cycle is often not in the kernel being
    validated. Any Gemini kernel containing a cycle is rejected at its own
    definition, so it can never become a callee -- but a kernel built with
    another dialect group (``squin.kernel``, say) runs no such guard, and a
    Gemini kernel may invoke one. Then the cycle sits one or more hops away and
    naming it alone would not say how it is reached.
    """

    route: tuple[ir.Method, ...]
    members: tuple[ir.Method, ...]


def format_cycle(cycle: Cycle | list[ir.Method]) -> str:
    """Render a cycle as ``a -> b -> a``, closing the loop for readability.

    A cycle reached from elsewhere also reports the route, as
    ``b -> b (reached via main -> a -> b)``.
    """
    if isinstance(cycle, Cycle):
        members, route = list(cycle.members), list(cycle.route)
    else:  # a bare list of members, for convenience at the call site
        members, route = list(cycle), []

    loop = " -> ".join([_name(method) for method in members] + [_name(members[0])])
    if not route:
        return loop

    via = " -> ".join(_name(method) for method in [*route, members[0]])
    return f"{loop} (reached via {via})"


@dataclass(init=False)
class CallGraph:
    """Static call graph rooted at ``entry``, keyed caller -> callees.

    Construction is an iterative worklist, so a recursive kernel produces a
    graph containing a cycle instead of exhausting the Python stack.

    Only statically resolved edges are followed. A dynamic ``func.Call`` through
    an SSA value is invisible here: resolving one needs the same constant
    propagation that diverges on cyclic input, so it cannot be used by a guard
    whose job is to run first.

    ``ir.Method.backedges`` looks like it should make this class unnecessary, but
    cannot be used for any of the three things needed here:

    - **It is empty when this runs.** ``update_backedges`` is called from
      ``Method.__init__``, which completes only *after* ``run_pass`` returns, so
      at guard time every method's ``backedges`` is still an empty set.
    - **It goes stale.** It is populated once at construction and never
      invalidated, so after the inliner has spliced a callee away the edge is
      still recorded. That matters for the post-fold re-check, which would then
      see cycles that no longer exist.
    - **It points the wrong way.** It records callers, but validating a kernel
      asks what that kernel *reaches*, rooted at the kernel itself.

    It also derives from the same ``ir.StaticCall`` trait, so it would miss the
    unresolved self call for the same reason a plain forward walk does.
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
                    continue

                if (
                    self_value is not None
                    and isinstance(stmt, func.Call)
                    and stmt.callee is self_value
                ):
                    # Not-yet-resolved self call; record it as a real self edge.
                    callees.add(caller)
                    continue

                for operand in _callable_operands(stmt):
                    if self_value is not None and operand is self_value:
                        callees.add(caller)
                    elif (callee := _constant_method(operand)) is not None:
                        callees.add(callee)

            self.edges[caller] = callees
            worklist.extend(callee for callee in callees if callee not in self.edges)

    def callees(self, method: ir.Method) -> list[ir.Method]:
        """Callees of ``method``, in a deterministic order."""
        return sorted(self.edges.get(method, ()), key=_sort_key)

    def _roots(self) -> list[ir.Method]:
        """Traversal roots, entry first.

        Entry leads so that every reachable cycle is discovered from the kernel
        being validated, which is what makes `Cycle.route` a path the reader can
        actually follow. The remaining roots are unreachable from entry in
        practice -- `__init__` only walks outwards from it -- and are kept as a
        guard against a hand-built `edges` dict.
        """
        rest = (m for m in sorted(self.edges, key=_sort_key) if m is not self.entry)
        return ([self.entry] if self.entry in self.edges else []) + list(rest)

    def find_cycles(self) -> list[Cycle]:
        """Return one representative cycle per distinct set of mutual callers.

        An empty list means the call graph is acyclic. The traversal is an
        explicit-stack DFS so that deeply nested (but acyclic) call graphs stay
        safe too.
        """
        cycles: list[Cycle] = []
        reported: set[frozenset[ir.Method]] = set()
        visited: set[ir.Method] = set()

        for root in self._roots():
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
                        # A back edge: `callee` is an ancestor of the node being
                        # expanded, so the walk has come back around to it. Cut
                        # the path at that ancestor -- from it onwards is the
                        # cycle, and everything before it is how `entry` reaches
                        # the cycle:
                        #
                        #     path = [main, a, b], callee = a
                        #       -> route   = (main,)
                        #       -> members = (a, b)   # b calls a, closing it
                        #
                        # Located by identity: `Method.__eq__` is generated by
                        # `@dataclass` while its `__hash__` is `id`, so `index`
                        # would compare fields where only identity is meant.
                        start = next(i for i, m in enumerate(path) if m is callee)
                        members = tuple(path[start:])

                        # Keyed on the member set, so one group of mutually
                        # recursive methods is reported once no matter how many
                        # back edges lead into it.
                        if (key := frozenset(members)) not in reported:
                            reported.add(key)
                            cycles.append(
                                Cycle(route=tuple(path[:start]), members=members)
                            )

                        # Recorded, never followed: descending into an ancestor
                        # is precisely what would loop forever.
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
        # NOTE: the caret is deliberately pinned to the validated kernel even
        # when the cycle is further down the call graph. `ValidationSuite` calls
        # `err.attach(method)` with the method it was handed, and `attach`
        # resolves source lines against *that* method's `py_func`, so blaming a
        # statement owned by a callee renders a line number from the wrong file.
        # `Cycle.route` carries the location information instead: the message
        # names the full path from this kernel down to the cycle.
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
    value is a hole in it: hand a kernel in as a value and call it and the result
    is a cycle no amount of graph walking will find, costing ``phi ** max_depth``
    in the address analysis once the branching factor reaches two. Resolving the
    target instead would need the constant propagation that diverges on exactly
    that input, so the shape is rejected rather than resolved.

    Both a ``func.Call`` and a higher-order statement's callable operand count:
    ``ilist.Map`` and friends invoke their ``fn`` without being a ``func.Call``.

    **Known gap.** Matching the parameter object is bypassed by routing the
    callable through anything at all before calling it -- an ``if``/``else`` makes
    it a block argument of a successor block, a list makes it a ``getitem``
    result -- and neither is the parameter, so neither is caught. Rejecting every
    unresolved call instead was tried and is not viable: after inlining, a
    perfectly ordinary stdlib kernel calls a self value belonging to an inlined
    region rather than to the method being scanned, so it is reported as opaque
    and the physical pipeline stops compiling. Closing this properly needs
    provenance for callable values, which is the analysis scoped in #927.

    Every statically reachable method is scanned, not just the entry. Checking
    only the entry leaves the hang reachable through a callee: a Gemini kernel
    can statically invoke a kernel from another dialect group, which never ran
    this guard and may make an opaque call of its own.
    """

    def name(self) -> str:
        return "gemini.no_opaque_call"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        graph = CallGraph(method)
        errors: list[ir.ValidationError] = []

        for caller in sorted(graph.edges, key=_sort_key):
            parameters = _parameter_values(caller)
            self_value = _self_value(caller)

            for stmt in caller.code.walk():
                # A `func.Call` on a parameter, and the same through a
                # higher-order statement's callable operand -- `ilist.Map` and
                # friends invoke their `fn` without being a `func.Call`.
                if isinstance(stmt, func.Call):
                    candidates = [stmt.callee]
                else:
                    candidates = _callable_operands(stmt)

                for callee in candidates:
                    if self_value is not None and callee is self_value:
                        # Known exactly; reported as a cycle instead.
                        continue

                    if callee not in parameters:
                        continue

                    name = f"the parameter '{callee.name or '<unnamed>'}'"
                    # Name the owner when it is not the kernel being validated,
                    # for the same reason `Cycle` carries a route: the caret is
                    # pinned to the entry method, so the message carries it.
                    owner = "" if caller is method else f" in '{_name(caller)}'"
                    errors.append(
                        ir.ValidationError(
                            method.code,
                            f"calling {name}{owner} is not supported in Gemini "
                            "kernels; the call target must be known at compile "
                            "time",
                        )
                    )

        return graph, errors


def check_call_graph(method: ir.Method) -> None:
    """Require ``method``'s call graph to be statically resolvable and acyclic.

    Raises ``ValidationErrorGroup`` listing every cycle and every opaque call
    found, rather than stopping at the first.

    Callers should run this *before* any pass that walks callees, and must not
    gate it on a `verify` flag. A cyclic call graph has no finite lowering, and
    the inliner and address analysis diverge on one instead of erroring, so
    skipping the check turns a clear rejection back into an unkillable hang
    (bloqade-lanes#921).
    """
    ValidationSuite([NoRecursionValidation, NoOpaqueCallValidation]).validate(
        method
    ).raise_if_invalid()
