from dataclasses import dataclass
from typing import Any, Callable, TypeGuard, TypeVar

from kirin import interp, ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import Encoder


@dataclass
class _ValidationAnalysis(Forward[EmptyLattice]):

    keys = "move.address.validation"

    arch_spec: ArchSpec

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return EmptyLattice.bottom()

    def eval_fallback(self, frame: ForwardFrame[EmptyLattice], node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)

    AddressType = TypeVar("AddressType", bound=Encoder)

    def filter_by_error(
        self,
        addresses: tuple[AddressType, ...],
        checker: Callable[[AddressType], str | None],
    ):
        """Apply a checker function to a sequence of addresses, yielding those with errors
        along with their error messages.

        Args:
            addresses: A tuple of address objects to be checked.
            checker: A function that takes an address and returns an error message
                if the address is invalid, or None if it is valid.
        Yields:
            Tuples of (address, error message) for each address that has an error.
        """

        def has_error(tup: tuple[Any, str | None]) -> TypeGuard[tuple[Any, str]]:
            return tup[1] is not None

        error_checks = zip(addresses, map(checker, addresses))
        yield from filter(has_error, error_checks)


@move.dialect.register(key="move.lane.validation")
class _MoveMethods(interp.MethodTable):
    @interp.impl(move.Move)
    def lane_checker(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.Move,
    ):
        if len(node.lanes) == 0:
            return ()

        invalid_lanes = []
        for lane, error_msg in _interp.filter_by_error(
            node.lanes, _interp.arch_spec.valid_lane
        ):
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Invalid lane address {lane!r}: {error_msg}",
                ),
            )

        valid_lanes = set(node.lanes) - set(invalid_lanes)
        first_lane = valid_lanes.pop()
        incompatible_lanes = []
        for other_lane in valid_lanes:
            if not _interp.arch_spec.compatible_lanes(first_lane, other_lane):
                incompatible_lanes.append(other_lane)

        if incompatible_lanes:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Move has incompatible lanes: {first_lane!r} and {invalid_lanes!r}",
                ),
            )

    @interp.impl(move.Initialize)
    @interp.impl(move.LocalR)
    @interp.impl(move.LocalRz)
    @interp.impl(move.Fill)
    def location_checker(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.Initialize | move.LocalR | move.LocalRz | move.Fill,
    ):

        invalid_locations = list(
            _interp.filter_by_error(
                node.location_addresses,
                _interp.arch_spec.valid_location,
            )
        )

        for lane_address, error_msg in invalid_locations:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Invalid location address {lane_address!r}: {error_msg}",
                ),
            )


@dataclass
class Validation(ValidationPass):
    """Validates a move program against an architecture specification."""

    arch_spec: ArchSpec

    def name(self) -> str:
        return "Lane Architecture Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:

        analysis = _ValidationAnalysis(
            method.dialects,
            arch_spec=self.arch_spec,
        )
        frame, _ = analysis.run(method)

        return frame, analysis.get_validation_errors()
