"""Per-statement validation for ``gemini.common.NewAt``.

Registered against the lanes validation interpreter key (``move.address.validation``).
The impl checks (1) const-foldability of the three SSA int args and (2) that the
resulting LocationAddress is valid for the architecture (via the existing
ArchSpec.check_location_group called by ``_ValidationAnalysis.report_location_errors``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kirin import interp, ir
from kirin.analysis.forward import ForwardFrame
from kirin.lattice.empty import EmptyLattice

from bloqade.gemini.common import dialect, stmts

if TYPE_CHECKING:
    from bloqade.lanes.validation.address import _ValidationAnalysis


@dialect.register(key="move.address.validation")
class _NewAtValidation(interp.MethodTable):
    @interp.impl(stmts.NewAt)
    def check_new_at(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: stmts.NewAt,
    ):
        # Lazy import to avoid circular initialisation:
        # any bloqade.lanes.* import triggers bloqade.lanes.__init__, which
        # imports bloqade.gemini.device → … → bloqade.gemini (partially
        # initialised at registration time).
        from bloqade.lanes.bytecode.encoding import LocationAddress

        z = _expect_const_int(node.zone_id, "zone_id", node, _interp)
        w = _expect_const_int(node.word_id, "word_id", node, _interp)
        s = _expect_const_int(node.site_id, "site_id", node, _interp)

        if z is None or w is None or s is None:
            return (EmptyLattice.bottom(),)

        candidate = LocationAddress(word_id=w, site_id=s, zone_id=z)
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
                "explicit allocation requires constant zone/word/site",
            ),
        )
    return data
