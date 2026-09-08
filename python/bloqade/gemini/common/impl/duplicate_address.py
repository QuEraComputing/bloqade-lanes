"""Cross-statement validation: each ``NewAt``'s address must be unique.

Implemented as a Forward dataflow analysis with a method-table impl for
``stmts.NewAt``. The impl pulls each coordinate via ``expect_const`` and
accumulates a seen-map on the interpreter. A second NewAt pinning the same
coordinate records a ``ValidationError``.

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
        _interp: _DuplicateAddressValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):
        z = _interp.expect_const(node.zone, int)
        r = _interp.expect_const(node.row, int)
        c = _interp.expect_const(node.col, int)

        coordinate = (z, r, c)
        if coordinate in _interp.seen:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"coordinate (zone={z}, row={r}, col={c}) is pinned by two "
                    f"operations.new_at calls",
                ),
            )
        else:
            _interp.seen[coordinate] = node

        return (EmptyLattice.bottom(),)
