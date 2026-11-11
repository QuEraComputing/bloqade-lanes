"""Arch-independent validation: every ``NewAt`` address arg must const-fold.

``NewAt`` pins a qubit to a physical address, so its three ``int`` arguments
have to be known at compile time. Nothing in ``verify()`` says so -- the
statement signature is satisfied by any ``int`` SSA value, including a kernel
block argument whose value only ever exists at runtime.

This is the arch-independent half of the check in :mod:`.new_at`, split out so
it can run at kernel-decoration time. Const-foldability needs only the
const-prop hints the unroll/fold passes have already attached; the in-range
half needs an ``ArchSpec``, which the gemini kernel groups do not have, and so
stays in :mod:`.new_at` for ``NativeToPlace.emit`` to run against a real spec.

Rejecting an entry kernel's own argument loses nothing: a parameter scan
compiles one program and binds arguments at runtime, so an entry argument can
never become a constant. An address threaded through a *callee* parameter is
unaffected -- it const-folds once the caller is inlined, which happens before
this pass runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kirin import interp, ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.dialects import arch


@dataclass
class _ConstAddressValidationAnalysis(Forward[EmptyLattice]):
    keys = ("lanes.arch.const_address",)
    lattice = EmptyLattice

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return self.lattice.bottom()

    def eval_fallback(self, frame: ForwardFrame[EmptyLattice], node: ir.Statement):
        return tuple(self.lattice.bottom() for _ in node.results)


@dataclass
class ConstAddressValidation(ValidationPass):
    """Report every ``gemini.common.NewAt`` address argument that is not a
    compile-time constant.
    """

    def name(self) -> str:
        return "lanes.arch.const_address"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        analysis = _ConstAddressValidationAnalysis(method.dialects)
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()


def _expect_const_int(
    value: ir.SSAValue,
    arg_name: str,
    node: ir.Statement,
    interpreter: Forward[EmptyLattice],
) -> int | None:
    """Read the const value for `value` via the AbstractInterpreter API. If
    absent or wrong type, emit a ValidationError on `node` naming the arg and
    return None.
    """
    data = interpreter.maybe_const(value, int)
    if data is None:
        interpreter.add_validation_error(
            node,
            ir.ValidationError(
                node,
                f"address argument '{arg_name}' is not a compile-time constant; "
                "explicit allocation requires constant zone/word/site",
            ),
        )
    return data


@arch.dialect.register(key="lanes.arch.const_address")
class _NewAtConstAddressMethods(interp.MethodTable):
    @interp.impl(arch.stmts.NewAt)
    def check_const_address(
        self,
        _interp: _ConstAddressValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: arch.stmts.NewAt,
    ):
        # Every arg is checked, not just the first bad one, so a kernel that
        # gets two of the three wrong hears about both in one pass.
        _expect_const_int(node.zone_id, "zone_id", node, _interp)
        _expect_const_int(node.word_id, "word_id", node, _interp)
        _expect_const_int(node.site_id, "site_id", node, _interp)

        return (EmptyLattice.bottom(),)
