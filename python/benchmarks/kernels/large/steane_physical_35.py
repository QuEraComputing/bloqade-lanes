"""Steane transversal 35-qubit demonstrator benchmark kernel."""

from kirin.dialects import ilist

from bloqade import squin


@squin.kernel(typeinfer=True, fold=True)
def steane_physical_35() -> None:
    """35-qubit physical circuit via transversal Steane logical CZs."""
    q = squin.qalloc(35)

    squin.broadcast.cz(
        ilist.IList(
            [
                q[0],
                q[1],
                q[2],
                q[3],
                q[4],
                q[5],
                q[6],
                q[14],
                q[15],
                q[16],
                q[17],
                q[18],
                q[19],
                q[20],
            ]
        ),
        ilist.IList(
            [
                q[7],
                q[8],
                q[9],
                q[10],
                q[11],
                q[12],
                q[13],
                q[21],
                q[22],
                q[23],
                q[24],
                q[25],
                q[26],
                q[27],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList(
            [
                q[7],
                q[8],
                q[9],
                q[10],
                q[11],
                q[12],
                q[13],
                q[21],
                q[22],
                q[23],
                q[24],
                q[25],
                q[26],
                q[27],
            ]
        ),
        ilist.IList(
            [
                q[14],
                q[15],
                q[16],
                q[17],
                q[18],
                q[19],
                q[20],
                q[28],
                q[29],
                q[30],
                q[31],
                q[32],
                q[33],
                q[34],
            ]
        ),
    )
    squin.broadcast.cz(
        ilist.IList([q[0], q[1], q[2], q[3], q[4], q[5], q[6]]),
        ilist.IList([q[28], q[29], q[30], q[31], q[32], q[33], q[34]]),
    )
