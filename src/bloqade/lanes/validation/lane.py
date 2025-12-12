from dataclasses import dataclass
from typing import Any

from kirin import interp, ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice.empty import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.dialects import move
from bloqade.lanes.layout.arch import ArchSpec


@dataclass
class _ValidationAnalysis(Forward[EmptyLattice]):

    keys = "move.lane.validation"

    arch_spec: ArchSpec

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return EmptyLattice.bottom()

    def eval_fallback(self, frame: ForwardFrame[EmptyLattice], node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)


@move.dialect.register(key="move.lane.validation")
class _MoveMethods(interp.MethodTable):
    @interp.impl(move.Move)
    def move(
        self,
        _interp: _ValidationAnalysis,
        frame: ForwardFrame[EmptyLattice],
        node: move.Move,
    ):
        if len(node.lanes) == 0:
            return ()

        lane = node.lanes[0]
        invalid_lanes = []
        for other_lane in node.lanes[1:]:
            if not _interp.arch_spec.compatible_lanes(lane, other_lane):
                invalid_lanes.append(other_lane)

        if invalid_lanes:
            _interp.add_validation_error(
                node,
                ir.ValidationError(
                    node,
                    f"Move has incompatible lanes: {lane!r} and {invalid_lanes!r}",
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
