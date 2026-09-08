"""Per-statement validation for ``gemini.common.NewAt``.

Registered against the lanes validation interpreter key (``move.address.validation``).
The impl checks (1) const-foldability of the three SSA int args and (2) that the
grid coordinate resolves to a valid location in the active architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kirin import interp, ir
from kirin.analysis.forward import ForwardFrame
from kirin.lattice.empty import EmptyLattice

from bloqade.gemini.common.dialects import qubit

if TYPE_CHECKING:
    from bloqade.lanes.validation.address import _ValidationAnalysis


@qubit.dialect.register(key="move.address.validation")
class _NewAtValidation(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def check_new_at(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):
        z = _expect_const_int(node.zone, "zone", node, _interp)
        r = _expect_const_int(node.row, "row", node, _interp)
        c = _expect_const_int(node.col, "col", node, _interp)

        if z is None or r is None or c is None:
            return (EmptyLattice.bottom(),)

        candidate = _interp.arch_spec.location_at(z, r, c)
        if candidate is None:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "Invalid location address: no location at "
                    f"(zone={z}, row={r}, col={c})",
                ),
            )
            return (EmptyLattice.bottom(),)

        _interp.report_location_errors(node, (candidate,))

        return (EmptyLattice.bottom(),)


def _expect_const_int(
    value: ir.SSAValue,
    arg_name: str,
    node: ir.Statement,
    interpreter: _ValidationAnalysis,
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
                "explicit allocation requires constant zone/row/col",
            ),
        )
    return data
