"""Eager per-statement validation for ``arrange.MoveTo`` (failures 1-5 from spec §4).

Registered against the lanes validation interpreter key (``move.address.validation``).
The impl checks:

1. ``len(qubits) == len(locations)``
2. ``locations`` must be compile-time constants (const-foldable)
3. Every ``LocationAddress`` is in-range for the arch spec
4. No duplicate destination ``LocationAddress`` values within one call
5. No duplicate Qubit SSA values within one call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kirin import interp, ir
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import ilist
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.gemini.common.dialects.arrange import dialect as arrange_dialect
from bloqade.gemini.common.dialects.arrange.stmts import MoveTo, Permute
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.dialects.arch import Loc

if TYPE_CHECKING:
    from bloqade.lanes.validation.address import _ValidationAnalysis


@arrange_dialect.register(key="move.address.validation")
class _MoveToValidationMethods(interp.MethodTable):
    @interp.impl(MoveTo)
    def check_move_to(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: MoveTo,
    ):
        # Lazy import to avoid circular initialisation
        from kirin.analysis import const

        # qubits must be an ilist.New result so we can count them;
        # ResultValue gives us the owning statement via .owner
        from kirin.ir import ResultValue

        from bloqade.lanes.bytecode.encoding import LocationAddress

        if not isinstance(node.qubits, ResultValue):
            _interp.add_validation_error(
                node,
                ir.ValidationError(node, "move_to: qubits must be a literal list"),
            )
            return (EmptyLattice.bottom(),)

        qubits_owner = node.qubits.owner
        if not isinstance(qubits_owner, ilist.New):
            _interp.add_validation_error(
                node,
                ir.ValidationError(node, "move_to: qubits must be a literal list"),
            )
            return (EmptyLattice.bottom(),)

        qubit_values = qubits_owner.values

        # Failure 2: locations must be compile-time constants
        locs_hint = node.locations.hints.get("const")
        if not isinstance(locs_hint, const.Value):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "move_to: locations must be compile-time constants "
                    "(pass a literal list of LocationAddress values)",
                ),
            )
            return (EmptyLattice.bottom(),)

        location_values: tuple[LocationAddress, ...] = tuple(locs_hint.data)

        # Failure 1: length mismatch
        if len(qubit_values) != len(location_values):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"move_to: len(qubits)={len(qubit_values)} != "
                    f"len(locations)={len(location_values)}",
                ),
            )
            return (EmptyLattice.bottom(),)

        # Failure 3: out-of-range LocationAddress (delegated to arch spec)
        _interp.report_location_errors(node, location_values)

        # Failure 4: duplicate destination addresses
        if len(set(location_values)) != len(location_values):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "move_to: duplicate destination LocationAddress within one call",
                ),
            )

        # Failure 5: duplicate qubit SSA values
        seen_ids: set[int] = set()
        for qv in qubit_values:
            if id(qv) in seen_ids:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        "move_to: same Qubit SSA value appears more than once "
                        "in qubits list",
                    ),
                )
                break
            seen_ids.add(id(qv))

        return (EmptyLattice.bottom(),)

    @interp.impl(Permute)
    def check_permute(
        self,
        _interp: "_ValidationAnalysis",
        frame: ForwardFrame[EmptyLattice],
        node: Permute,
    ):
        from kirin.analysis import const
        from kirin.ir import ResultValue

        if not isinstance(node.qubits, ResultValue) or not isinstance(
            qubits_owner := node.qubits.owner, ilist.New
        ):
            _interp.add_validation_error(
                node,
                ir.ValidationError(node, "permute: qubits must be a literal list"),
            )
            return (EmptyLattice.bottom(),)

        qubit_values = qubits_owner.values

        # P1: perm must be compile-time constant
        perm_hint = node.perm.hints.get("const")
        if not isinstance(perm_hint, const.Value):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "permute: perm must be compile-time constants "
                    "(pass a literal list of ints)",
                ),
            )
            return (EmptyLattice.bottom(),)

        perm_values = tuple(int(p) for p in perm_hint.data)

        # P2: length mismatch
        if len(qubit_values) != len(perm_values):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"permute: len(qubits)={len(qubit_values)} != "
                    f"len(perm)={len(perm_values)}",
                ),
            )
            return (EmptyLattice.bottom(),)

        # P3: perm must be a bijection of range(n)
        if sorted(perm_values) != list(range(len(perm_values))):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    "permute: perm must be a permutation of range(len(qubits)) "
                    "(each index 0..n-1 exactly once)",
                ),
            )

        # P4: duplicate qubit SSA values
        seen_ids: set[int] = set()
        for qv in qubit_values:
            if id(qv) in seen_ids:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        "permute: same Qubit SSA value appears more than once "
                        "in qubits list",
                    ),
                )
                break
            seen_ids.add(id(qv))

        return (EmptyLattice.bottom(),)

    @interp.impl(Loc)
    def check_loc(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: Loc,
    ):
        from bloqade.gemini.common.validation.new_at import _expect_const_int
        from bloqade.lanes.bytecode.encoding import LocationAddress

        z = _expect_const_int(node.zone_id, "zone_id", node, _interp)
        w = _expect_const_int(node.word_id, "word_id", node, _interp)
        s = _expect_const_int(node.site_id, "site_id", node, _interp)

        if z is None or w is None or s is None:
            return (EmptyLattice.bottom(),)

        _interp.report_location_errors(
            node, (LocationAddress(zone_id=z, word_id=w, site_id=s),)
        )
        return (EmptyLattice.bottom(),)


@dataclass
class MoveToValidation(ValidationPass):
    """Eager per-statement validation for ``arrange.MoveTo`` (failures 1-5)."""

    arch_spec: ArchSpec = field(kw_only=True)

    def name(self) -> str:
        return "lanes.move_to.validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        from bloqade.lanes.validation.address import _ValidationAnalysis

        analysis = _ValidationAnalysis(
            method.dialects,
            arch_spec=self.arch_spec,
        )
        frame, _ = analysis.run(method)
        return frame, analysis.get_validation_errors()
