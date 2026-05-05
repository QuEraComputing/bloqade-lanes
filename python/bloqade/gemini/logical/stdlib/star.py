from kirin import lowering

from bloqade import types

from ..dialects.operations.stmts import DEFAULT_STEANE_STAR_SUPPORT, StarRz


@lowering.wraps(StarRz)
def star_rz(
    theta: float,
    qubit: types.Qubit,
    qubit_indices: tuple[int, int, int] = DEFAULT_STEANE_STAR_SUPPORT,
) -> None: ...
