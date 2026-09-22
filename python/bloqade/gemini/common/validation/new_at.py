"""Per-statement validation for ``gemini.common.NewAt``.

Registered against the lanes validation interpreter key (``move.address.validation``).
The impl checks (1) const-foldability of the three SSA int args and (2) that the
resulting LocationAddress is valid for the architecture (via the existing
ArchSpec.check_location_group called by ``_ValidationAnalysis.report_location_errors``).

Check (1) is arch-independent and also runs on its own, at kernel-decoration
time, via :mod:`.const_address` -- this module is where it runs for callers that
already hold an ``ArchSpec`` and want (2) as well. Re-running it here is what
lets a kernel compiled without the decorator's validation suite still get the
const diagnostic instead of a range check on a garbage address.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kirin import interp
from kirin.analysis.forward import ForwardFrame
from kirin.lattice.empty import EmptyLattice

from bloqade.gemini.common.dialects import qubit

from .const_address import _expect_const_int

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
