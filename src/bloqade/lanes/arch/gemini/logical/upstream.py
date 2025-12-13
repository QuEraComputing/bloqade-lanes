from dataclasses import replace
from typing import Iterator, TypeVar

from kirin import ir, rewrite

from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress
from bloqade.lanes.rewrite.transversal import (
    RewriteGetMeasurementResult,
    RewriteLogicalToPhysicalConversion,
)

from .rewrite import RewriteFill, RewriteInitialize, RewriteMoves
from .stmts import dialect

AddressType = TypeVar("AddressType", bound=LocationAddress | LaneAddress)


def steane7_transversal_map(address: AddressType) -> Iterator[AddressType] | None:
    """This function is used to map logical addresses to physical addresses.

    The Steane [[7,1,3]] code encodes one logical qubit into seven physical qubits.
    The mapping is as follows:

    Logical Word ID 0 -> Physical Word IDs 0 to 6
    Logical Word ID 1 -> Physical Word IDs 8 to 14

    All other Word IDs remain unchanged.

    """
    if address.word_id == 0:
        return (replace(address, word_id=word_id) for word_id in range(7))
    elif address.word_id == 1:
        return (replace(address, word_id=word_id) for word_id in range(8, 15, 1))
    else:
        return None

class SpecializeGemini:

    def emit(self, mt: ir.Method, no_raise=True) -> ir.Method:
        out = mt.similar(dialects=mt.dialects.add(dialect))

        rewrite.Walk(
            rewrite.Chain(
                RewriteMoves(),
                RewriteFill(),
                RewriteInitialize(),
                RewriteGetMeasurementResult(steane7_transversal_map),
                RewriteLogicalToPhysicalConversion(),
            )
        ).rewrite(out.code)

        if not no_raise:
            out.verify()

        return out
