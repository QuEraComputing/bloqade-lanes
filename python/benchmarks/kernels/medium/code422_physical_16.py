"""Four pinned [[4,2,2]] blocks, two per word: a routing baseline.

Every qubit is pinned, so the initial layout is fixed and the benchmark measures
placement and routing only. Each block sits on contiguous sites of one word with
position ``p`` at site ``offset + p``:

    b0: word 0, sites 0-3    b1: word 0, sites 4-7
    b2: word 2, sites 0-3    b3: word 2, sites 4-7

No code blocks are registered. A block-aware placement strategy can register
the same blocks on the same pins and be compared against these rows.
"""

from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini.common.dialects import qubit
from bloqade.gemini.common.dialects.qubit import new_at

pinned_kernel = squin.kernel.add(qubit)


@pinned_kernel(typeinfer=True, fold=True)
def code422_physical_16():
    b0 = ilist.IList(
        [new_at(0, 0, 0), new_at(0, 0, 1), new_at(0, 0, 2), new_at(0, 0, 3)]
    )
    b1 = ilist.IList(
        [new_at(0, 0, 4), new_at(0, 0, 5), new_at(0, 0, 6), new_at(0, 0, 7)]
    )
    b2 = ilist.IList(
        [new_at(0, 2, 0), new_at(0, 2, 1), new_at(0, 2, 2), new_at(0, 2, 3)]
    )
    b3 = ilist.IList(
        [new_at(0, 2, 4), new_at(0, 2, 5), new_at(0, 2, 6), new_at(0, 2, 7)]
    )

    # Encode every block into logical |00>: H then a CX fan-out from position 0.
    heads = ilist.IList([b0[0], b1[0], b2[0], b3[0]])
    squin.broadcast.h(heads)
    for p in range(1, 4):
        squin.broadcast.cx(heads, ilist.IList([b0[p], b1[p], b2[p], b3[p]]))

    # Transversal CZ over every block pairing: same-word pairs and cross-word pairs.
    squin.broadcast.cz(b0 + b2, b1 + b3)
    squin.broadcast.cz(b0 + b1, b2 + b3)
    squin.broadcast.cz(b0 + b1, b3 + b2)
