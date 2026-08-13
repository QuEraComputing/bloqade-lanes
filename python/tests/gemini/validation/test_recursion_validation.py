"""Recursive Gemini kernels must be rejected, and rejected *fast*.

Before the guard in `logical.kernel`, a cyclic call graph did not raise -- it
sent `AddressAnalysis` into a phi**max_depth traversal that never returned
(bloqade-lanes#921). Some tests here would hang rather than fail if the guard
regressed, which is why they assert on elapsed time as well as on the error.

Note that recursion reaches the guard in two different IR shapes. A kernel
calling itself is lowered as a *dynamic* call on its own `%<name>_self` block
argument, and is only rewritten into a static `func.Invoke` after `run_pass`
returns; a call into an already-built kernel is a static `func.Invoke` straight
away. `CallGraph` has to understand both, so both are covered below.
"""

import time

import bloqade.squin as squin
import pytest
from kirin.ir.exception import ValidationErrorGroup
from kirin.lowering.exception import BuildError
from kirin.validation import ValidationSuite

import bloqade.gemini as gemini
from bloqade.gemini.common.validation.recursion import (
    CallGraph,
    NoRecursionValidation,
    format_cycle,
)

# Generous enough to absorb a slow CI box, tight enough that the pre-fix
# behaviour (effectively unbounded) cannot pass.
REJECTION_BUDGET_S = 60.0


def _messages(err: ValidationErrorGroup) -> list[str]:
    return [e.args[0] if e.args else str(e) for e in err.errors]


def test_direct_self_recursion_is_rejected():
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def recurse(q):
            recurse(q)

    messages = _messages(exc_info.value)
    assert any("recursion is not supported" in m for m in messages)
    assert any("recurse -> recurse" in m for m in messages)


def test_verify_false_still_rejects_recursion():
    """`verify=False` must not be able to skip the guard.

    The address analysis that diverges runs unconditionally in `run_pass`, so a
    check gated on `verify` would leave the hang reachable.
    """
    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(verify=False, no_raise=True)
        def recurse(q):
            recurse(q)

    assert time.monotonic() - started < REJECTION_BUDGET_S


def test_aggressive_unroll_still_rejects_recursion():
    with pytest.raises(ValidationErrorGroup):

        @gemini.logical.kernel(aggressive_unroll=True)
        def recurse(q):
            recurse(q)


def test_original_hang_reproducer_is_rejected_quickly():
    """The mwe from bloqade-lanes#921: `main -> B`, `B -> {B, main}`.

    This is the Fibonacci-shaped graph whose cost grew as phi**depth. `B` alone
    is enough to reject: its `B(main)` is a static self edge and its `main()` is
    a call through a parameter, so both rules fire and `main` never gets built.
    """

    @gemini.logical.kernel(verify=False)
    def simple(reg):
        squin.broadcast.h(reg)

    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def B(main):
            B(main)
            main()

    assert time.monotonic() - started < REJECTION_BUDGET_S
    assert any("B -> B" in m for m in _messages(exc_info.value))


def test_mutual_recursion_between_kernels_cannot_be_lowered():
    """Two kernels calling each other is not expressible, let alone compilable.

    The first kernel's forward reference fails during lowering, so the guard
    never sees it. Pinned here so that a future lowering change which *does*
    admit the forward reference shows up as a failure needing a guard update.
    """

    with pytest.raises(BuildError):

        @gemini.logical.kernel(verify=False)
        def ping(q):
            pong(q)  # pyright: ignore[reportUndefinedVariable]  # noqa: F821


def test_calling_a_parameter_is_rejected():
    """A call through a parameter has no statically knowable target.

    `CallGraph` cannot see such an edge, so it is rejected by shape rather than
    resolved -- resolving it would need the constant propagation that diverges
    on the cyclic inputs this guard exists to catch.
    """
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def higher_order(f):
            f()

    assert any(
        "calling the parameter 'f' is not supported" in m
        for m in _messages(exc_info.value)
    )


def test_dynamic_cycle_with_branching_cannot_be_built():
    """The shape that used to hang with no static edge anywhere.

    Previously `main` passed itself to `B`, which only ever called it
    dynamically; two dynamic calls give the cycle branching factor 2, so the
    address analysis cost was phi**max_depth while `CallGraph` saw no edge at
    all. `B` is now rejected at its own definition, so the cycle can never be
    assembled -- which is what makes that hang unreachable rather than merely
    less likely.
    """
    started = time.monotonic()
    with pytest.raises(ValidationErrorGroup) as exc_info:

        @gemini.logical.kernel(verify=False)
        def B(f):
            f()
            f()

    assert time.monotonic() - started < REJECTION_BUDGET_S
    # Reported once per offending call site, not once per kernel.
    messages = _messages(exc_info.value)
    assert sum("calling the parameter 'f'" in m for m in messages) == 2


def test_non_recursive_subkernel_calls_are_unaffected():
    @gemini.logical.kernel
    def flip(q):
        squin.x(q)

    @gemini.logical.kernel
    def main():
        q = squin.qalloc(2)
        flip(q[0])
        flip(q[1])

    main.print()


def test_diamond_call_graph_is_not_a_cycle():
    """Two paths to the same callee is sharing, not recursion."""

    @gemini.logical.kernel
    def leaf(q):
        squin.x(q)

    @gemini.logical.kernel
    def left(q):
        leaf(q)

    @gemini.logical.kernel
    def right(q):
        leaf(q)

    @gemini.logical.kernel
    def main():
        q = squin.qalloc(2)
        left(q[0])
        right(q[1])

    assert CallGraph(main).find_cycles() == []


def test_callgraph_records_edges_caller_to_callee():
    @gemini.logical.kernel
    def leaf(q):
        squin.x(q)

    # inline=False keeps the `func.Invoke` in place instead of splicing the
    # callee body in; verify=False because a surviving invoke is separately
    # rejected by the logical validation.
    @gemini.logical.kernel(verify=False, inline=False)
    def caller():
        q = squin.qalloc(1)
        leaf(q[0])

    graph = CallGraph(caller)
    assert graph.entry is caller

    callees = graph.callees(caller)
    assert leaf in callees
    # Stdlib kernels are ordinary edges too -- `squin.qalloc` is itself a kernel.
    assert {m.sym_name for m in callees} == {"leaf", "qalloc"}
    # `callees` is sorted, so diagnostics are stable across runs.
    assert [m.sym_name for m in callees] == ["leaf", "qalloc"]
    assert graph.find_cycles() == []


def test_callgraph_terminates_and_finds_multi_node_cycles():
    """`kirin.analysis.CallGraph` raises RecursionError on cyclic input (kirin#703).

    A multi-kernel cycle cannot be built through the decorator (see
    `test_mutual_recursion_between_kernels_cannot_be_lowered`), so the traversal
    is exercised directly on an injected cycle.
    """

    @gemini.logical.kernel
    def a(q):
        squin.x(q)

    @gemini.logical.kernel
    def b(q):
        squin.y(q)

    @gemini.logical.kernel
    def c(q):
        squin.z(q)

    graph = CallGraph(a)
    graph.edges = {a: {b}, b: {c}, c: {a}}

    cycles = graph.find_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {a, b, c}
    # Reported as a rotation starting from the DFS root, with the loop closed.
    assert format_cycle(cycles[0]).split(" -> ")[0] == "a"
    assert format_cycle(cycles[0]).count("->") == 3


def test_callgraph_reports_one_cycle_per_group():
    @gemini.logical.kernel
    def a(q):
        squin.x(q)

    @gemini.logical.kernel
    def b(q):
        squin.y(q)

    graph = CallGraph(a)
    # `a` self-loops and also sits in a 2-cycle with `b`.
    graph.edges = {a: {a, b}, b: {a}}

    cycles = graph.find_cycles()
    assert [set(cycle) for cycle in cycles] == [{a}, {a, b}]


def test_no_recursion_validation_pass_reports_no_errors_for_acyclic_kernel():
    @gemini.logical.kernel
    def main():
        q = squin.qalloc(2)
        squin.h(q[0])

    ValidationSuite([NoRecursionValidation]).validate(main).raise_if_invalid()


def test_format_cycle_closes_the_loop():
    @gemini.logical.kernel
    def a(q):
        squin.x(q)

    @gemini.logical.kernel
    def b(q):
        squin.y(q)

    assert format_cycle([a]) == "a -> a"
    assert format_cycle([a, b]) == "a -> b -> a"
