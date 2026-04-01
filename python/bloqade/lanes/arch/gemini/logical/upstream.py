from typing import Iterator, Literal, TypeVar

from kirin import ir, rewrite
from kirin.dialects import debug, ilist

from bloqade import qubit, squin
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress
from bloqade.lanes.rewrite.transversal import (
    RewriteGetItem,
    RewriteLogicalToPhysicalConversion,
)

from .rewrite import RewriteFill, RewriteInitialize, RewriteMoves
from .stmts import dialect

AddressType = TypeVar("AddressType", bound=LocationAddress | LaneAddress)


def steane7_transversal_map(address: AddressType) -> Iterator[AddressType] | None:
    """Map logical addresses to physical addresses via site expansion.

    The Steane [[7,1,3]] code encodes one logical qubit into seven physical qubits.
    Each logical site expands to 7 physical sites within the same word:

        Logical site s → Physical sites s*7, s*7+1, ..., s*7+6

    Word ID is preserved. Works universally for any word.
    """
    base = address.site_id * 7
    return (address.replace(site_id=base + i) for i in range(7))


@squin.kernel
def steane7_initialize(
    theta: float, phi: float, lam: float, qubits: ilist.IList[qubit.Qubit, Literal[7]]
):
    debug.info("Begin Steane7 Initialize")
    squin.u3(theta, phi, lam, qubits[6])
    squin.broadcast.sqrt_y_adj(qubits[:6])
    evens = qubits[::2]  # [0, 2, 4, 6]
    odds = qubits[1::2]  # [1, 3, 5]

    # Fixed: CZ pairs should be (1,2), (3,4), (5,6) not (1,0), (3,2), (5,4)
    squin.broadcast.cz(odds, evens[1:])
    squin.sqrt_y(qubits[6])
    squin.broadcast.cz(evens[:-1], ilist.IList([qubits[3], qubits[5], qubits[6]]))
    squin.broadcast.sqrt_y(qubits[2:])
    squin.broadcast.cz(evens[:-1], odds)
    squin.broadcast.sqrt_y(ilist.IList([qubits[1], qubits[2], qubits[4]]))
    squin.x(qubits[3])
    squin.broadcast.z(ilist.IList([qubits[1], qubits[5]]))
    debug.info("End Steane7 Initialize")


class SpecializeGemini:

    def __init__(self, sites_per_word: int = 16):
        self.sites_per_word = sites_per_word

    def emit(self, mt: ir.Method, no_raise=True) -> ir.Method:
        out = mt.similar(dialects=mt.dialects.add(dialect))

        rewrite.Walk(
            rewrite.Chain(
                RewriteMoves(sites_per_word=self.sites_per_word),
                RewriteFill(),
                RewriteInitialize(),
                RewriteGetItem(steane7_transversal_map),
                RewriteLogicalToPhysicalConversion(),
            )
        ).rewrite(out.code)

        if not no_raise:
            out.verify()

        return out
