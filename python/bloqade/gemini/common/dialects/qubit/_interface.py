from bloqade.types import Qubit
from kirin import lowering

from .stmts import NewAt


@lowering.wraps(NewAt)
def new_at(zone: int, row: int, col: int) -> Qubit:
    """Allocate a qubit at a ``(zone, row, col)`` grid coordinate.

    ``col`` is the grid x-index and ``row`` the grid y-index within ``zone``.
    Compilation resolves the coordinate to the architecture's internal
    ``(word_id, site_id)`` address. All three arguments must be compile-time
    constants.
    """
    ...
