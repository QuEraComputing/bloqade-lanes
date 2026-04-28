from bloqade.types import Qubit
from kirin import lowering

from .stmts import NewAt


@lowering.wraps(NewAt)
def new_at(zone_id: int, word_id: int, site_id: int) -> Qubit:
    """Allocate a qubit pinned to (zone_id, word_id, site_id).

    All three arguments must be compile-time constants. Use of non-constant
    values raises a validation error before lowering.
    """
    ...
