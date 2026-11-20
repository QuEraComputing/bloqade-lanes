from typing import Any

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from bloqade.lanes.dialects.steane713.extensions.stmts import StarR, StarRz
from bloqade.lanes.dialects.steane713.star import DEFAULT_STEANE_STAR_SUPPORT


@lowering.wraps(StarRz)
def star_rz(
    rotation_angle: float,
    qubits: ilist.IList[Qubit, Any],
    qubit_indices: tuple[int, int, int] = DEFAULT_STEANE_STAR_SUPPORT,
) -> None:
    """Apply a STAR/TMR logical-Z rotation injection primitive.

    Args:
        rotation_angle: Target logical rotation angle in SQuIn IR turn units.
        qubits: List of logical qubits.
        qubit_indices: Steane weight-3 logical-Z support. Defaults to ``(4, 5, 6)``.
    """
    ...


@lowering.wraps(StarR)
def star_r(
    axis_angle: float,
    rotation_angle: float,
    qubits: ilist.IList[Qubit, Any],
    qubit_indices: tuple[int, int, int] = DEFAULT_STEANE_STAR_SUPPORT,
) -> None:
    """Apply a STAR/TMR logical XY-plane rotation injection primitive.

    Same argument order and names as native ``r``. STAR-X is
    ``star_r(0.0, theta, qubits)``.

    Args:
        axis_angle: Logical rotation axis in the XY plane, in SQuIn IR turns.
            Any value; quarter turns (0, 1/4, 1/2, 3/4) lower most directly.
        rotation_angle: Target logical rotation angle in SQuIn IR turns.
        qubits: List of logical qubits.
        qubit_indices: Steane weight-3 logical support (keyword only in
            kernels, since it is an attribute). Defaults to ``(4, 5, 6)``.
    """
    ...
