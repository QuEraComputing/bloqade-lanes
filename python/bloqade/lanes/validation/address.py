from dataclasses import dataclass, field
from itertools import chain
from typing import Any

from kirin import interp, ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress


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
    @interp.impl(move.Fill)
    def location_checker(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.LogicalInitialize | move.LocalR | move.LocalRz | move.Fill,
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


@dataclass
class Validation(ValidationPass):
    """Validates a move program against an architecture specification."""

    arch_spec: ArchSpec = field(kw_only=True)

    def name(self) -> str:
        return "lanes.address.validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:

        analysis = _ValidationAnalysis(
            method.dialects,
            arch_spec=self.arch_spec,
        )
        frame, _ = analysis.run(method)

        return frame, analysis.get_validation_errors()
