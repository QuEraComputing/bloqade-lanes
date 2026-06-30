from typing import TypeVar, overload

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from bloqade.lanes.bytecode.encoding import LocationAddress

from .stmts import CzPartner, Loc, MoveTo, Permute


@lowering.wraps(Loc)
def loc(zone_id: int, word_id: int, site_id: int) -> LocationAddress:
    """Construct a LocationAddress for use with move_to inside a kernel body.

    All three arguments must be compile-time-constant integers.
    """
    ...


@lowering.wraps(CzPartner)
def cz_partner(address: LocationAddress) -> LocationAddress:
    """Return the CZ blockade-partner location of ``address``.

    An atom placed at the returned location can be CZ-entangled with an atom
    at ``address``. Useful for staging a ``move_to`` onto a partner site
    without hardcoding the arch's word/site layout, e.g.::

        movement.move_to([q], [movement.cz_partner(static_loc)])

    ``address`` must resolve to a compile-time-constant location (the partner
    is looked up in the architecture spec during compilation).
    """
    ...


Len = TypeVar("Len")


@overload
def move_to(
    qubits: ilist.IList[Qubit, Len],
    locations: ilist.IList[LocationAddress, Len],
    multi_move_warning: bool = True,
): ...


@overload
def move_to(
    qubits: list[Qubit],
    locations: list[LocationAddress],
    multi_move_warning: bool = True,
): ...


@lowering.wraps(MoveTo)
def move_to(
    qubits,
    locations,
    multi_move_warning=True,
) -> None:
    """Move the given qubits to the specified LocationAddress destinations.

    User-directed movement within a single zone; the compiler tracks where the
    atoms end up, so a ``move_to`` may be followed by single-qubit gates,
    another ``move_to``, a CZ, or a terminal measurement. (Under palindrome
    return moves the atoms are returned to their pre-move layout at the next CZ,
    so in that mode a ``move_to`` must reach a CZ before any measurement.)
    Compilation fails if the move is physically infeasible.
    """
    ...


@overload
def permute(
    qubits: ilist.IList[Qubit, Len],
    perm: ilist.IList[int, Len],
): ...


@overload
def permute(
    qubits: list[Qubit],
    perm: list[int],
): ...


@lowering.wraps(Permute)
def permute(
    qubits,
    perm,
) -> None:
    """Move qubits into a permutation of their own current locations.

    ``qubits[i]`` moves to the location currently held by ``qubits[perm[i]]``.
    ``perm`` must be a compile-time-constant permutation of
    ``range(len(qubits))``. Like ``move_to``, this is user-directed movement
    within an inter-CZ segment.
    """
    ...
