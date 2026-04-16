"""Reproducible GHZ q4 benchmark kernel."""

from kirin.dialects import ilist

from bloqade import qubit, squin


@squin.kernel(typeinfer=True, fold=True)
def ghz_4():
    """4-qubit GHZ state using the log-depth doubling pattern."""
    size = 4
    q0 = qubit.new()
    squin.h(q0)
    reg = ilist.IList([q0])
    for _ in range(size):
        current = len(reg)
        missing = size - current
        if missing > current:
            num_alloc = current
        else:
            num_alloc = missing
        if num_alloc > 0:
            new_qubits = qubit.qalloc(num_alloc)
            squin.broadcast.cx(reg[-num_alloc:], new_qubits)
            reg = reg + new_qubits
