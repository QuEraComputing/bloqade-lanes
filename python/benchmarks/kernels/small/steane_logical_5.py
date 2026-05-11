"""Steane logical 5-qubit demonstrator benchmark kernel."""

from kirin.dialects import ilist

from bloqade import squin


@squin.kernel(typeinfer=True, fold=True)
def steane_logical_5() -> None:
    """5-qubit logical circuit with layered CZ interactions."""
    logical = squin.qalloc(5)

    squin.broadcast.cz(
        ilist.IList([logical[0], logical[2]]),
        ilist.IList([logical[1], logical[3]]),
    )
    squin.broadcast.cz(
        ilist.IList([logical[1], logical[3]]),
        ilist.IList([logical[2], logical[4]]),
    )
    squin.broadcast.cz(ilist.IList([logical[0]]), ilist.IList([logical[4]]))
