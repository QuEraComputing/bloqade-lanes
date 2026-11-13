"""Cross-statement validation: each ``NewAt``'s address must be unique.

Implemented as a Forward dataflow analysis with a method-table impl for
``stmts.NewAt``. The impl pulls each address arg via ``maybe_const``, builds a
``LocationAddress``, and accumulates a seen-map on the interpreter. A second
NewAt pinning the same address records a ``ValidationError``.

A non-const arg is skipped, not reported: const-foldability belongs to
``ConstAddressValidation``, which runs alongside this pass. Skipping keeps the
two independent -- a kernel with both a non-const address and a genuine
duplicate gets both diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kirin import interp, ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import arch as qubit
from bloqade.lanes.dialects.arch.stmts import NewAt


@dataclass
class _DuplicateAddressValidationAnalysis(Forward[EmptyLattice]):
    keys = ("lanes.arch.duplicates",)
    lattice = EmptyLattice

    seen: dict[LocationAddress, NewAt] = field(init=False, default_factory=dict)

    def initialize(self):
        self.seen.clear()
        return super().initialize()

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return self.lattice.bottom()

    def eval_fallback(self, frame: ForwardFrame[EmptyLattice], node: ir.Statement):
        return tuple(self.lattice.bottom() for _ in node.results)


@dataclass
class DuplicateAddressValidation(ValidationPass):
    """Report any pair of ``lanes.arch.NewAt`` statements that pin the
    same physical address.
    """

    def name(self) -> str:
        return "lanes.arch.duplicates"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        analysis = _DuplicateAddressValidationAnalysis(method.dialects)
        # `run`, not `run_no_raise`: nothing in this analysis raises by design,
        # and swallowing one that does would truncate the walk and drop every
        # duplicate after it -- silently, which is the failure mode this pass
        # exists to prevent. ValidationSuite already turns an unexpected raise
        # into a reported error rather than a crash.
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()


@qubit.dialect.register(key="lanes.arch.duplicates")
class _NewAtDuplicateMethods(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def check_duplicate(
        self,
        _interp: _DuplicateAddressValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):

        z = _interp.maybe_const(node.zone_id, int)
        w = _interp.maybe_const(node.word_id, int)
        s = _interp.maybe_const(node.site_id, int)

        if z is None or w is None or s is None:
            # No address to compare, so nothing to say -- ConstAddressValidation
            # reports the non-const arg. Skip this statement rather than raise:
            # raising aborts the whole forward walk, which used to make one
            # non-const NewAt silently hide every duplicate after it.
            return (EmptyLattice.bottom(),)

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
