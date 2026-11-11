from collections.abc import Iterable
from typing import cast

DEFAULT_STEANE_STAR_SUPPORT = (4, 5, 6)
VALID_STEANE_STAR_SUPPORTS = frozenset(
    {
        (0, 1, 5),
        (0, 2, 4),
        (0, 3, 6),
        (1, 2, 6),
        (1, 3, 4),
        (2, 3, 5),
        DEFAULT_STEANE_STAR_SUPPORT,
    }
)


def validate_steane_star_support(
    qubit_indices: Iterable[int] | None,
) -> tuple[int, int, int]:
    support = DEFAULT_STEANE_STAR_SUPPORT if qubit_indices is None else qubit_indices
    out = tuple(support)
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in out):
        raise ValueError("qubit_indices must contain integer physical qubit indices")
    if out not in VALID_STEANE_STAR_SUPPORTS:
        valid = ", ".join(
            str(support) for support in sorted(VALID_STEANE_STAR_SUPPORTS)
        )
        raise ValueError(
            f"qubit_indices must be a valid Steane weight-3 logical-Z support; "
            f"got {out}. Valid Steane supports are: {valid}"
        )
    return cast(tuple[int, int, int], out)
