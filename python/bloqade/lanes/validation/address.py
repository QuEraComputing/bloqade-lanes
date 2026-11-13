from dataclasses import dataclass
from itertools import chain
from typing import Any, ClassVar

from kirin import interp, ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.dialects import ilist
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.dialects import arch as qubit, move
from bloqade.lanes.dialects.arch import Loc
from bloqade.lanes.dialects.arrange import dialect as arrange_dialect
from bloqade.lanes.dialects.arrange.stmts import MoveTo, Permute
from bloqade.lanes.validation.const_address import _expect_const_int


@dataclass
class _ValidationAnalysis(Forward[EmptyLattice]):
    lattice = EmptyLattice
    keys = ("move.address.validation",)

    arch_spec: ArchSpec

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return EmptyLattice.bottom()

    def eval_fallback(self, frame: ForwardFrame[EmptyLattice], node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)

    def report_location_errors(
        self, node: ir.Statement, locations: tuple[LocationAddress, ...]
    ):
        """Validate a group of locations via Rust and report errors."""
        for error in self.arch_spec.check_location_group(locations):
            self.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Invalid location address: {error}",
                ),
            )

    def report_lane_errors(self, node: ir.Statement, lanes: tuple[LaneAddress, ...]):
        """Validate a group of lanes via Rust and report errors."""
        for error in self.arch_spec.check_lane_group(lanes):
            self.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Invalid lane group (count={len(lanes)}): {error}",
                ),
            )


@move.dialect.register(key="move.address.validation")
class _MoveMethods(interp.MethodTable):
    @interp.impl(move.Move)
    def lane_checker(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.Move,
    ):
        if len(node.lanes) > 0:
            _interp.report_lane_errors(node, node.lanes)
        return (EmptyLattice.bottom(),)

    @interp.impl(move.LogicalInitialize)
    @interp.impl(move.LocalR)
    @interp.impl(move.LocalRz)
    @interp.impl(move.StarRz)
    @interp.impl(move.StarRx)
    @interp.impl(move.Fill)
    def location_checker(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: (
            move.LogicalInitialize
            | move.LocalR
            | move.LocalRz
            | move.StarRz
            | move.StarRx
            | move.Fill
        ),
    ):
        _interp.report_location_errors(node, node.location_addresses)
        return (EmptyLattice.bottom(),)

    @interp.impl(move.GetFutureResult)
    def location_checker_get_future(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.GetFutureResult,
    ):
        _interp.report_location_errors(node, (node.location_address,))

    @interp.impl(move.PhysicalInitialize)
    def location_checker_physical(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.PhysicalInitialize,
    ):
        all_locations = tuple(chain.from_iterable(node.location_addresses))
        _interp.report_location_errors(node, all_locations)


def get_validation(arch_spec: ArchSpec) -> type[ValidationPass]:
    """Build an address-validation pass bound to ``arch_spec``.

    ``kirin``'s :class:`ValidationSuite` instantiates each pass with no
    arguments (``pass_cls()``), so a pass cannot take ``arch_spec`` as a
    constructor argument. We close over ``arch_spec`` and stash it on a
    :data:`ClassVar`, yielding a no-arg-constructible pass class that the
    suite can drive directly.
    """

    @dataclass
    class Validation(ValidationPass):
        """Validates a move program against an architecture specification."""

        ARCH_SPEC: ClassVar[ArchSpec] = arch_spec

        def name(self) -> str:
            return "lanes.address.validation"

        def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
            analysis = _ValidationAnalysis(
                method.dialects,
                arch_spec=self.ARCH_SPEC,
            )
            frame, _ = analysis.run(method)

            return frame, analysis.get_validation_errors()

    return Validation


@qubit.dialect.register(key="move.address.validation")
class _NewAtValidation(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def check_new_at(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: qubit.stmts.NewAt,
    ):
        z = _expect_const_int(node.zone_id, "zone_id", node, _interp)
        w = _expect_const_int(node.word_id, "word_id", node, _interp)
        s = _expect_const_int(node.site_id, "site_id", node, _interp)

        if z is None or w is None or s is None:
            return (EmptyLattice.bottom(),)

        candidate = LocationAddress(word_id=w, site_id=s, zone_id=z)
        _interp.report_location_errors(node, (candidate,))

        return (EmptyLattice.bottom(),)


@arrange_dialect.register(key="move.address.validation")
class _MoveToValidationMethods(interp.MethodTable):
    @interp.impl(MoveTo)
    def check_move_to(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: MoveTo,
    ):
        # qubits must be an ilist.New result so we can count them;
        # ResultValue gives us the owning statement via .owner
        from kirin.analysis import const
        from kirin.ir import ResultValue

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
        _interp: _ValidationAnalysis,
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
        """Validate a residual (non-const-folded) ``arch.loc(zone, row, col)``.

        A constant ``loc`` const-folds to a ``LocationAddress`` during
        ``AggressiveUnroll`` and never reaches validation; this fires only on a
        residual ``Loc`` — coordinates that are not compile-time constants, or
        (defensively) that don't resolve to a location in the arch spec. The
        ``(zone, row, col)`` resolution against ``ArchSpec.location_at`` is the
        in-range check for the coordinate addressing scheme.
        """

        zone = _expect_const_int(node.zone, "zone", node, _interp)
        row = _expect_const_int(node.row, "row", node, _interp)
        col = _expect_const_int(node.col, "col", node, _interp)

        if zone is None or row is None or col is None:
            return (EmptyLattice.bottom(),)

        if _interp.arch_spec.location_at(zone, row, col) is None:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"loc: no location at (zone={zone}, row={row}, col={col}) "
                    "in the architecture spec",
                ),
            )

        return (EmptyLattice.bottom(),)
