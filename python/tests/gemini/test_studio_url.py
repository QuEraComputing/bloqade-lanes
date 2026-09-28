"""The Gemini Studio link exporter preserves the supported logical circuit."""

import pytest
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import logical


@logical.kernel(aggressive_unroll=True)
def bell():
    qubits = squin.qalloc(2)
    squin.h(qubits[0])
    squin.cx(qubits[0], qubits[1])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def broadcast():
    qubits = squin.qalloc(4)
    squin.broadcast.h(qubits)
    squin.broadcast.cz(qubits[0:2], qubits[2:4])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def pinned():
    qubits = logical.qalloc_at(ilist.IList([3, None, 1, None]))
    squin.cz(qubits[0], qubits[2])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def prepared():
    qubits = squin.qalloc(1)
    squin.u3(0.25, 0.5, 0.75, qubits[0])
    squin.s_adj(qubits[0])
    return logical.default_post_processing(qubits)


@logical.kernel(aggressive_unroll=True)
def unsupported_t():
    qubits = squin.qalloc(1)
    squin.t(qubits[0])
    return logical.default_post_processing(qubits)


def test_bell_link() -> None:
    assert logical.to_studio_url(bell) == (
        "https://bloqade.quera.com/studio/gemini/#circuit="
        "{version:13,qubits:2,gates:[{gate:h,column:0,targets:[0]},"
        "{gate:cx,column:3,targets:[0,1]}]}"
    )


def test_broadcast_gates_share_columns() -> None:
    url = logical.to_studio_url(broadcast)
    assert url.count("gate:h,column:0") == 4
    assert "{gate:cz,column:3,targets:[0,2]}" in url
    assert "{gate:cz,column:3,targets:[1,3]}" in url


def test_pinned_slots_and_unpinned_conflict() -> None:
    url = logical.to_studio_url(pinned)
    assert "{gate:cz,column:0,targets:[0,2]}" in url
    assert "placement:[{qubit:0,slot:3},{qubit:1,slot:0}," in url
    assert "{qubit:2,slot:1},{qubit:3,slot:2}]" in url


def test_initial_state_and_adjoint_gate() -> None:
    url = logical.to_studio_url(prepared)
    assert "{gate:s_adj,column:0,targets:[0]}" in url
    assert "prep:[{qubit:0,theta:0.25,phi:0.5,lam:0.75}]" in url


def test_unsupported_nonclifford_does_not_disappear() -> None:
    with pytest.raises(ValueError, match="cannot represent.*T"):
        logical.to_studio_url(unsupported_t)
