"""Canonical Pauli-string values for Lanes APIs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar, Union


class Pauli(str, Enum):
    """A single-qubit Pauli operator."""

    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"

    @classmethod
    def coerce(cls, value: Pauli | str) -> Pauli:
        """Return the Pauli represented by ``value``.

        Args:
            value: A :class:`Pauli` or one-character ``"I"``, ``"X"``,
                ``"Y"``, or ``"Z"`` label.

        Raises:
            TypeError: If ``value`` is not a Pauli or string.
            ValueError: If ``value`` is not a valid Pauli label.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"Pauli must be a Pauli or string, got {type(value).__name__}"
            )
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Invalid Pauli {value!r}") from exc


PauliTerm = tuple[int, Pauli]
SparsePauliLike = Mapping[int, Pauli | str]
PauliStringLike = Union["PauliString", str, SparsePauliLike, Sequence[Pauli | str]]
_ValueT = TypeVar("_ValueT")


@dataclass(frozen=True)
class PauliString:
    """An immutable sparse Pauli string over a fixed number of qubits.

    ``terms`` stores only non-identity Paulis as sorted ``(qubit, pauli)``
    pairs. Dense labels use increasing qubit index from left to right, so
    ``"XIZ"`` means X on qubit 0 and Z on qubit 2.
    """

    num_qubits: int
    terms: tuple[PauliTerm, ...]

    def __post_init__(self) -> None:
        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")

        normalized: dict[int, Pauli] = {}
        for term in self.terms:
            try:
                qubit, pauli = term
            except (TypeError, ValueError) as exc:
                raise TypeError("Each Pauli term must be an (int, Pauli) pair") from exc
            if isinstance(qubit, bool) or not isinstance(qubit, int):
                raise TypeError("Pauli qubit indices must be integers")
            if qubit < 0:
                raise ValueError("Pauli qubit indices must be non-negative")
            if qubit >= self.num_qubits:
                raise ValueError("num_qubits is too small for Pauli qubit index")
            normalized_pauli = Pauli.coerce(pauli)
            if normalized_pauli is Pauli.I:
                continue
            if qubit in normalized:
                raise ValueError(f"Duplicate Pauli term for qubit {qubit}")
            normalized[qubit] = normalized_pauli

        object.__setattr__(self, "terms", tuple(sorted(normalized.items())))

    @classmethod
    def from_dense(cls, label: str) -> PauliString:
        """Create a Pauli string from a dense, left-to-right qubit label."""
        if not isinstance(label, str):
            raise TypeError("Dense Pauli labels must be strings")
        return cls(
            num_qubits=len(label),
            terms=tuple(
                (qubit, pauli)
                for qubit, pauli in enumerate(map(Pauli.coerce, label))
                if pauli is not Pauli.I
            ),
        )

    @classmethod
    def from_sparse(
        cls,
        terms: SparsePauliLike,
        *,
        num_qubits: int | None = None,
    ) -> PauliString:
        """Create a Pauli string from a sparse qubit-to-Pauli mapping."""
        if not isinstance(terms, Mapping):
            raise TypeError("Sparse Pauli terms must be a mapping")
        normalized = tuple(
            (qubit, Pauli.coerce(pauli)) for qubit, pauli in terms.items()
        )
        if num_qubits is None:
            num_qubits = max((qubit for qubit, _ in normalized), default=-1) + 1
        return cls(num_qubits=num_qubits, terms=normalized)

    @classmethod
    def coerce(cls, value: object) -> PauliString:
        """Convert a supported dense or sparse representation to ``PauliString``."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_dense(value)
        if isinstance(value, Mapping):
            return cls.from_sparse(value)
        if isinstance(value, Sequence):
            return cls.from_dense("".join(Pauli.coerce(pauli).value for pauli in value))
        raise TypeError(
            "PauliString must be a PauliString, dense string, sparse mapping, or sequence"
        )

    def __getitem__(self, qubit: int) -> Pauli:
        """Return the Pauli at ``qubit``, or identity when it is not in ``terms``."""
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError("Pauli qubit indices must be integers")
        if not 0 <= qubit < self.num_qubits:
            raise IndexError("Pauli qubit index is out of range")
        return dict(self.terms).get(qubit, Pauli.I)

    @property
    def support(self) -> tuple[int, ...]:
        """Qubit indices with a non-identity Pauli."""
        return tuple(qubit for qubit, _ in self.terms)

    def __str__(self) -> str:
        """Return the dense Pauli label in increasing qubit-index order."""
        label = [Pauli.I.value] * self.num_qubits
        for qubit, pauli in self.terms:
            label[qubit] = pauli.value
        return "".join(label)


class PauliMapping(Mapping[PauliString, _ValueT], Generic[_ValueT]):
    """Read-only mapping that normalizes Pauli-string keys at its boundary."""

    def __init__(
        self,
        entries: Mapping[Any, _ValueT] | Iterable[tuple[object, _ValueT]] = (),
    ) -> None:
        data: dict[PauliString, _ValueT] = {}
        iterator = entries.items() if isinstance(entries, Mapping) else entries
        for key, value in iterator:
            pauli_string = PauliString.coerce(key)
            if pauli_string in data:
                raise ValueError(f"Duplicate PauliString key {pauli_string!s}")
            data[pauli_string] = value
        self._data = data

    def __getitem__(self, key: PauliStringLike) -> _ValueT:
        return self._data[PauliString.coerce(key)]

    def __iter__(self) -> Iterator[PauliString]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        try:
            return PauliString.coerce(key) in self._data
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


__all__ = ["Pauli", "PauliMapping", "PauliString", "PauliStringLike"]
