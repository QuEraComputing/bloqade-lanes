from typing import TypeVar, overload

from bloqade.types import Qubit
from kirin import lowering
from kirin.dialects import ilist

from bloqade.lanes.bytecode.encoding import LocationAddress

from .stmts import MoveTo, Permute

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
    insert_moves: bool = False,
): ...


@overload
def permute(
    qubits: list[Qubit],
    perm: list[int],
    insert_moves: bool = False,
): ...


@lowering.wraps(Permute)
def permute(
    qubits,
    perm,
    insert_moves=False,
) -> None:
    """Permute qubits — a logical relabel where ``qubits[i]`` ends up referring
    to what was ``qubits[perm[i]]``.

    ``perm`` must be a compile-time-constant permutation of ``range(len(qubits))``.
    ``insert_moves`` (compile-time constant) selects how the permutation is
    realized:

    - ``False`` (default): **relabel only** — no atoms move; the qubit references
      are permuted (the quantum information is permuted for free). A later
      transversal move absorbs the physical permutation (a "lazy" permutation).
      Valid under any strategy, including palindrome.
    - ``True``: **also commit the moves** — physically route the atoms so the
      permutation is realized in place now (e.g. to keep a QEC code block
      physically arranged to match the new labels). Only valid under a
      non-palindrome (no-return) strategy; ``PalindromePlacementStrategy``
      rejects it, since it would return the committed moves home.
    """
    ...
