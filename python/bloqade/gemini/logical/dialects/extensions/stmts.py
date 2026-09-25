from bloqade.types import QubitType
from kirin import exception, ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.gemini.star import (
    DEFAULT_STEANE_STAR_SUPPORT,
    validate_steane_star_support,
)

from ._dialect import dialect


@statement(dialect=dialect)
class StarRz(ir.Statement):
    """STAR/TMR logical-Z rotation injection primitive.

    ``rotation_angle`` is the target logical angle in SQuIn IR turn units.
    ``qubits`` is an ``IList`` of logical qubits.
    The final logical-to-physical lowering computes the physical STAR angle for
    k=3 and emits physical Rz rotations on ``qubit_indices``.
    """

    traits = frozenset({lowering.FromPythonCall()})
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
