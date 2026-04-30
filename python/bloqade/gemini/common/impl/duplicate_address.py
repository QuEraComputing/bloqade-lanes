"""Cross-statement validation: each ``NewAt``'s address must be unique.

Implemented as a Forward dataflow analysis with a method-table impl for
``stmts.NewAt``. The impl pulls each address arg via ``expect_const``, builds a
``LocationAddress``, and accumulates a seen-map on the interpreter. A second
NewAt pinning the same address records a ``ValidationError``.

Per-statement validation (const-foldability + range) is the precondition;
when an arg is non-const, ``expect_const`` raises ``InterpreterError``. The
``ValidationPass`` wrapper uses ``run_no_raise`` so any duplicates collected
before that point are still reported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kirin import interp, ir
from kirin.analysis.forward import ForwardFrame
from kirin.lattice.empty import EmptyLattice

from ..dialects import qubit

if TYPE_CHECKING:
    from ..validation.duplicate_address import (
        _DuplicateAddressValidationAnalysis,
    )


@qubit.dialect.register(key="gemini.common.qubit.duplicates")
class _NewAtDuplicateMethods(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def check_duplicate(
        self,
        _interp: "_DuplicateAddressValidationAnalysis",
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):
        from bloqade.lanes.bytecode.encoding import LocationAddress

        z = _interp.expect_const(node.zone_id, int)
        w = _interp.expect_const(node.word_id, int)
        s = _interp.expect_const(node.site_id, int)

        addr = LocationAddress(word_id=w, site_id=s, zone_id=z)
        if addr in _interp.seen:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"address (zone={z}, word={w}, site={s}) is pinned by two "
                    f"operations.new_at calls",
                ),
            )
        else:
            _interp.seen[addr] = node

        return (EmptyLattice.bottom(),)
