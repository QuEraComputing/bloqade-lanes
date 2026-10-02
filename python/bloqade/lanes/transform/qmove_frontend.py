"""Lower a kernel and its subroutines to native IR, keeping subroutine calls.

Everything except the listed subroutines is inlined, without unrolling. Each
method is processed as its own ``similar()`` clone rather than through
``CallGraphPass`` / ``SquinToNative.emit``: those clone every callee and retarget
the calls, after which a subroutine call no longer references the user's
``Method`` and the subroutine would be inlined like any other call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

from bloqade.native._prelude import kernel as native_kernel
from bloqade.native.upstream.squin2native import GateRule
from bloqade.rewrite.passes.callgraph import ReplaceMethods
from kirin import ir, passes, rewrite
from kirin.ir.exception import ValidationErrorGroup
from kirin.rewrite import Inline

from bloqade.gemini.common.validation.recursion import CallGraph, format_cycle
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.dialects.arch import BindArchSpec
from bloqade.lanes.rewrite import clifford2native
from bloqade.lanes.rewrite.const_call_to_invoke import ConstCallToInvoke
from bloqade.lanes.rewrite.refine_qubit_types import RefineQubitTypes


@dataclass(frozen=True)
class NativeProgram:
    entry: ir.Method
    subroutines: dict[ir.Method, ir.Method]
    """Each original subroutine ``Method`` mapped to its lowered clone."""


def unlisted_recursion(
    entry: ir.Method, subroutines: frozenset[ir.Method]
) -> list[str]:
    """Call-graph cycles through a kernel that is not a subroutine.

    Checked before inlining: inlining a recursive kernel never reaches a fixpoint.
    """
    return [
        format_cycle(cycle)
        for cycle in CallGraph(entry).find_cycles()
        if any(member not in subroutines for member in cycle.members)
    ]


def lower_to_native(
    entry: ir.Method,
    subroutines: Iterable[ir.Method],
    arch_spec: ArchSpec,
    *,
    no_raise: bool = False,
) -> NativeProgram:
    subs = tuple(dict.fromkeys(subroutines))
    sub_codes = tuple(sub.code for sub in subs)
    reachable = CallGraph(entry).edges.keys()
    dialects = (
        entry.dialects.union(chain.from_iterable(m.dialects.data for m in reachable))
        .union(native_kernel)
        .add(qmove.dialect)
        .add(move.dialect)
    )

    def keep_call(code: ir.Statement) -> bool:
        # rewrite.Inline hands its heuristic the callee's func.Function, not the
        # call site.
        return all(code is not sub_code for sub_code in sub_codes)

    inline = rewrite.Fixpoint(
        rewrite.Walk(rewrite.Chain(ConstCallToInvoke(), Inline(keep_call)))
    )

    def inline_calls(method: ir.Method) -> None:
        # A Fixpoint that hits its iteration limit leaves calls un-inlined, and
        # such a call would sit off the state chain: a silent miscompile.
        if inline.rewrite(method.code).exceeded_max_iter:
            raise ValidationErrorGroup(
                "NativeToQMove: inlining did not converge",
                errors=[
                    ir.ValidationError(
                        method.code,
                        f"inlining did not converge after {inline.max_iter} "
                        f"iterations in {method.sym_name}; the call chain is "
                        "too deep or recursive",
                    )
                ],
            )

    def lower(method: ir.Method) -> ir.Method:
        out = method.similar(dialects)
        inline_calls(out)
        rewrite.Walk(BindArchSpec(arch_spec)).rewrite(out.code)
        rewrite.Walk(clifford2native.DecomposeCliffordToNative()).rewrite(out.code)
        # GateRule turns each squin gate into a call of a native stdlib kernel;
        # inline again to expose the native.gate statements.
        rewrite.Walk(GateRule()).rewrite(out.code)
        inline_calls(out)
        rewrite.Fixpoint(rewrite.Walk(rewrite.DeadCodeElimination())).rewrite(out.code)
        return out

    clones = {sub: lower(sub) for sub in subs}
    new_entry = lower(entry)
    for method in (new_entry, *clones.values()):
        rewrite.Walk(ReplaceMethods(clones)).rewrite(method.code)
        passes.TypeInfer(method.dialects, no_raise=no_raise)(method)
        RefineQubitTypes(method.dialects, no_raise=no_raise)(method)
    return NativeProgram(new_entry, clones)
