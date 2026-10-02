"""Narrow qubit-valued SSA types using qubit address analysis.

Type inference cannot recover qubit types once calls are inlined without
unrolling:

* ``func.invoke`` is typed from the callee's signature. Specializing it to the
  call's arguments would mutate a signature every call site shares, so
  ``qalloc(3)`` is typed ``IList[Qubit, Any]``.
* ``py.constant`` is typed ``PyClass(type(value))`` (``!py.int``), not a
  ``Literal``; values live in the const-prop lattice instead.
* Re-inference cannot narrow an inlined type: ``TypeInference.eval_fallback``
  substitutes solved type variables into the result's *current* SSA type, and
  an inlined statement arrives already concrete (``IList[Qubit, Any]``).

``AddressAnalysis`` re-interprets each callee with its real arguments, so it does
know ``qalloc(3)`` is a three-qubit register. This pass meets each qubit-valued
SSA type with the type its address implies. It only ever narrows; a contradiction
(a value typed ``Qubit`` that holds a register) is an error. Values type
inference already typed ``Bottom`` are left alone.

``TypeInfer`` overwrites every type it infers, so run this after each
``TypeInfer``, once per method: ``AddressAnalysis`` discards callee frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.analysis import address
from bloqade.types import QubitType
from kirin import ir, types
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup
from kirin.passes import Pass
from kirin.rewrite.abc import RewriteResult


def address_type(addr: address.Address) -> types.TypeAttribute | None:
    """The type an address implies, or ``None`` if it implies nothing."""
    if isinstance(addr, (address.AddressQubit, address.UnknownQubit)):
        return QubitType
    if isinstance(addr, address.AddressReg):
        return ilist.IListType[QubitType, types.Literal(len(addr.data))]
    if isinstance(addr, address.UnknownReg):
        return ilist.IListType[QubitType, types.Any]
    if isinstance(addr, address.PartialIList) and addr.data:
        elems = [address_type(elem) for elem in addr.data]
        if any(elem is None for elem in elems):
            return None
        joined = elems[0]
        assert joined is not None
        for elem in elems[1:]:
            assert elem is not None
            joined = joined.join(elem)
        return ilist.IListType[joined, types.Literal(len(elems))]
    return None


@dataclass
class RefineQubitTypes(Pass):
    def unsafe_run(self, mt: ir.Method) -> RewriteResult:
        analysis = address.AddressAnalysis(self.dialects)
        if self.no_raise:
            frame, _ = analysis.run_no_raise(mt)
        else:
            frame, _ = analysis.run(mt)

        errors: list[ir.ValidationError] = []
        changed = False
        for value, addr in frame.entries.items():
            derived = address_type(addr)
            if derived is None:
                continue
            current = value.type
            if current.is_subseteq(types.Bottom):
                # Type inference already rejected this value. Its address is
                # derived from that Bottom type (Bottom is a subtype of every
                # IList[Qubit]), so it says nothing about what the value holds.
                continue
            refined = current.meet(derived)
            if refined.is_subseteq(types.Bottom):
                node = value.owner if isinstance(value, ir.ResultValue) else mt.code
                errors.append(
                    ir.ValidationError(
                        node,
                        f"value typed {current} holds {addr}, which needs {derived}",
                    )
                )
                continue
            if refined != current:
                value.type = refined
                changed = True

        if errors and not self.no_raise:
            for error in errors:
                error.attach(mt)
            raise ValidationErrorGroup(
                "RefineQubitTypes: qubit types contradict their addresses",
                errors=errors,
            )
        return RewriteResult(has_done_something=changed)
