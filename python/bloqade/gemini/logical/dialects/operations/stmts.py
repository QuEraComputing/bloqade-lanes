import ast
from dataclasses import dataclass
from typing import Protocol, cast

from bloqade.types import MeasurementResultType, QubitType
from kirin import exception, ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist
from kirin.lowering import Result, State
from kirin.lowering.exception import BuildError

from bloqade.gemini.star import (
    DEFAULT_STEANE_STAR_SUPPORT,
    validate_steane_star_support,
)

from ._dialect import dialect


class _StarRzConstructor(Protocol):
    def __call__(
        self,
        rotation_angle: ir.SSAValue,
        qubits: ir.SSAValue,
        *,
        qubit_indices: tuple[int, ...] = DEFAULT_STEANE_STAR_SUPPORT,
    ) -> "StarRz": ...


@dataclass(frozen=True)
class FromPythonStarRzCall(lowering.FromPythonCall["StarRz"]):
    def lower(
        self, stmt: type["StarRz"], state: State[ast.AST], node: ast.Call
    ) -> Result:
        if len(node.args) > 3:
            raise BuildError("star_rz expects at most 3 positional arguments")

        args, kwargs = self.lower_Call_inputs(stmt, state, node)
        rotation_angle = cast(ir.SSAValue, args["rotation_angle"])
        qubits = cast(ir.SSAValue, args["qubits"])
        if not qubits.type.is_subseteq(ilist.IListType[QubitType, types.Any]):
            qubits = state.current_frame.push(
                ilist.New([qubits], elem_type=QubitType)
            ).result

        if len(node.args) == 3:
            if "qubit_indices" in kwargs:
                raise BuildError(
                    "qubit_indices was provided as both a positional and keyword argument"
                )
            kwargs["qubit_indices"] = tuple(
                state.get_global(node.args[2]).expect(tuple)
            )

        return state.current_frame.push(
            cast(_StarRzConstructor, stmt)(
                rotation_angle,
                qubits,
                **kwargs,
            )
        )


@statement(dialect=dialect)
class Initialize(ir.Statement):
    """Initialize a list of logical qubits to an arbitrary state.

    Args:
        phi (float): Angle for rotation around the Z axis
        theta (float): angle for rotation around the Y axis
        phi (float): angle for rotation around the Z axis
        qubits (IList[QubitType, Len]): The list of logical qubits to initialize

    """

    traits = frozenset({})
    theta: ir.SSAValue = info.argument(types.Float)
    phi: ir.SSAValue = info.argument(types.Float)
    lam: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, types.Any])


Len = types.TypeVar("Len")


@statement(dialect=dialect)
class TerminalLogicalMeasurement(ir.Statement):
    """Perform measurements on a list of logical qubits.

    Measurements are returned as a nested list where each member list
    contains the individual measurement results for the constituent physical qubits per logical qubit.

    Args:
        qubits (IList[QubitType, Len]): The list of logical qubits

    Returns:
        IList[IList[MeasurementResultType, CodeN], Len]: A nested list containing the measurement results,
            where each inner list corresponds to the measurements of the physical qubits that make up each logical qubit.
    """

    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, Len])
    num_physical_qubits: int | None = info.attribute(
        types.Int, init=False, default_factory=lambda: None
    )
    result: ir.ResultValue = info.result(
        ilist.IListType[ilist.IListType[MeasurementResultType, types.Any], Len]
    )


@statement(dialect=dialect, init=False)
class StarRz(ir.Statement):
    """STAR/TMR logical-Z rotation injection primitive.

    ``rotation_angle`` is the target logical angle. ``qubits`` is an ``IList``
    of logical qubits, matching the raw Squin single-qubit gate statements.
    The public stdlib API provides a one-qubit helper plus a broadcast helper.
    The final logical-to-physical lowering computes the physical STAR angle for
    k=3 and emits physical Rz rotations on ``qubit_indices``.
    """

    traits = frozenset({FromPythonStarRzCall()})
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(ilist.IListType[QubitType, types.Any])
    qubit_indices: tuple[int, int, int] = info.attribute(
        default=DEFAULT_STEANE_STAR_SUPPORT
    )

    def check(self) -> None:
        try:
            validate_steane_star_support(self.qubit_indices)
        except ValueError as exc:
            raise exception.StaticCheckError(str(exc)) from exc
