from __future__ import annotations

import pytest

from bloqade.lanes.pauli import Pauli, PauliMapping, PauliString


def test_pauli_string_canonicalizes_dense_and_sparse_representations():
    dense = PauliString.coerce("XIZ")
    sparse = PauliString.from_sparse({0: "X", 2: Pauli.Z}, num_qubits=3)

    assert dense == sparse
    assert dense.num_qubits == 3
    assert dense.terms == ((0, Pauli.X), (2, Pauli.Z))
    assert str(dense) == "XIZ"
    assert hash(dense) == hash(sparse)


def test_pauli_string_rejects_invalid_dense_and_sparse_terms():
    with pytest.raises(ValueError, match="Invalid Pauli"):
        PauliString.coerce("XA")

    with pytest.raises(ValueError, match="non-negative"):
        PauliString.from_sparse({-1: "X"})

    with pytest.raises(ValueError, match="too small"):
        PauliString.from_sparse({2: "X"}, num_qubits=2)


def test_pauli_mapping_normalizes_keys_on_construction_and_lookup():
    mapping = PauliMapping(
        [
            ("X", "x-kernel"),
            ({2: "Z"}, "z-kernel"),
        ]
    )

    assert mapping["X"] == "x-kernel"
    assert mapping[PauliString.coerce("X")] == "x-kernel"
    assert mapping[{2: Pauli.Z}] == "z-kernel"
    assert {str(key) for key in mapping} == {"X", "IIZ"}
    assert "X" in mapping
    assert {2: "Z"} in mapping


def test_pauli_mapping_rejects_duplicate_canonical_keys():
    with pytest.raises(ValueError, match="Duplicate PauliString"):
        PauliMapping([("X", 1), (PauliString.coerce("X"), 2)])
