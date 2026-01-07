from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable, Iterable, TypeVar

from kirin import interp, ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import Encoder, LaneAddress


@dataclass
class _ValidationAnalysis(Forward[EmptyLattice]):
    lattice = EmptyLattice
    keys = ("move.address.validation",)

    arch_spec: ArchSpec

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return EmptyLattice.bottom()

    def eval_fallback(self, frame: ForwardFrame[EmptyLattice], node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)

    AddressType = TypeVar("AddressType", bound=Encoder)

    def filter_by_error(
        self,
        addresses: Iterable[AddressType],
        checker: Callable[[AddressType], set[str]],
    ):
        """Apply a checker function to a sequence of addresses, yielding those with errors
        along with their error messages.

        Args:
            addresses: A tuple of address objects to be checked.
            checker: A function that takes an address and returns a set of error messages.
                if the set is empty, the address is considered valid.
        Yields:
            Tuples of (address, error message) for each address that has an error.
        """

        def has_error(tup: tuple[Any, set[str]]) -> bool:
            return len(tup[1]) > 0

        error_checks = zip(addresses, map(checker, addresses))
        yield from filter(has_error, error_checks)


@move.dialect.register(key="move.address.validation")
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
        for lane, error_msgs in _interp.filter_by_error(
            node.lanes, _interp.arch_spec.validate_lane
        ):
            invalid_lanes.append(lane)
            for error_msg in error_msgs:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        f"Invalid lane address {lane!r}: {error_msg}",
                    ),
                )

        valid_lanes = set(node.lanes) - set(invalid_lanes)
        if len(valid_lanes) == 0:
            return

        first_lane = valid_lanes.pop()
        incompatible_lanes = []

        def validate_compatible_lane(lane: LaneAddress):
            return _interp.arch_spec.compatible_lane_error(first_lane, lane)

        for lane, error_msgs in _interp.filter_by_error(
            valid_lanes, validate_compatible_lane
        ):
            incompatible_lanes.append(lane)
            for error_msg in error_msgs:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        f"Incompatible lane address {first_lane!r} with lane {lane!r}: {error_msg}",
                    ),
                )

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
        invalid_locations = list(
            _interp.filter_by_error(
                node.location_addresses,
                _interp.arch_spec.validate_location,
            )
        )

        for lane_address, error_msgs in invalid_locations:
            for error_msg in error_msgs:
                _interp.add_validation_error(
                    node,
                    ir.ValidationError(
                        node,
                        f"Invalid location address {lane_address!r}: {error_msg}",
                    ),
                )

        return (EmptyLattice.bottom(),)

    @interp.impl(move.GetFutureResult)
    def location_checker_get_future(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.GetFutureResult,
    ):
        location_address = node.location_address
        error_msgs = _interp.arch_spec.validate_location(location_address)

        for error_msg in error_msgs:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Invalid location address {location_address!r}: {error_msg}",
                ),
            )

    @interp.impl(move.PhysicalInitialize)
    def location_checker_physical(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.PhysicalInitialize,
    ):
        invalid_locations = list(
            _interp.filter_by_error(
                chain.from_iterable(node.location_addresses),
                _interp.arch_spec.validate_location,
            )
        )

        for lane_address, error_msgs in invalid_locations:
            for error_msg in error_msgs:
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

    arch_spec: ArchSpec = field(kw_only=True)

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:

        analysis = _ValidationAnalysis(
            method.dialects,
            arch_spec=self.arch_spec,
        )
        frame, _ = analysis.run(method)

        return frame, analysis.get_validation_errors()
