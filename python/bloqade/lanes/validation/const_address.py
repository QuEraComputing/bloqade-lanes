"""Arch-independent validation: every ``NewAt`` address arg must const-fold.

``NewAt`` pins a qubit to a physical address, so its three ``int`` arguments
have to be known at compile time. Nothing in ``verify()`` says so -- the
statement signature is satisfied by any ``int`` SSA value, including a kernel
block argument whose value only ever exists at runtime.

Gemini registers the per-statement method tables for this analysis in
``bloqade.gemini.common.impl.const_address``. The corresponding in-range
check uses ``ArchSpec`` during compilation and is registered by
``bloqade.gemini.common.impl.new_at``.

Rejecting an entry kernel's own argument loses nothing: a parameter scan
compiles one program and binds arguments at runtime, so an entry argument can
never become a constant. An address threaded through a *callee* parameter is
unaffected -- it const-folds once the caller is inlined, which happens before
this pass runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kirin import ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass


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
    """Report every ``lanes.arch.NewAt`` address argument that is not a
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
