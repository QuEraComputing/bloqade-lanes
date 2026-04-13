"""Foundation base classes for thin Python wrappers over PyO3 Rust objects.

PyO3 ``#[pyclass]`` types are final from Python and cannot subclass
abstract base classes that use a non-default metaclass (notably Kirin's
``ir.Attribute`` via ``ABCMeta``). Any Rust type that needs to participate
in the IR therefore needs a Python adapter.

This module provides two layered base classes that standardise that adapter
pattern, eliminate per-module wrap/unwrap helpers, and offer a zero
re-allocation fast path for converting a Rust result into a wrapper:

* :class:`RustWrapper` — minimal Python adapter with a fast
  :meth:`from_inner` classmethod and equality / hashing delegated to the
  underlying Rust object. No Kirin dependency.
* :class:`KirinRustWrapper` — extends :class:`RustWrapper` and additionally
  implements the :class:`kirin.ir.Data` contract (``type`` attribute,
  :meth:`unwrap`, :meth:`encode`, :meth:`print_impl`).

See issue #466 for design rationale.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from kirin import ir, types
from kirin.print import Printer
from typing_extensions import Self

R = TypeVar("R")


class RustWrapper(Generic[R]):
    """Thin Python adapter over a PyO3 Rust object.

    Subclasses set ``self._inner`` in ``__init__`` (typically by constructing
    the underlying Rust type) and may add Python-specific properties or
    methods. Hash and equality delegate to the Rust object so wrappers behave
    consistently with Rust semantics.

    The :meth:`from_inner` classmethod wraps an existing Rust object without
    re-allocating. Use it whenever the Rust object already exists — for
    example, when wrapping the result of another Rust call:

    .. code-block:: python

        rust_src, rust_dst = arch._inner.lane_endpoints(lane._inner)
        return LocationAddress.from_inner(rust_src), LocationAddress.from_inner(rust_dst)
    """

    _inner: R

    @classmethod
    def from_inner(cls, inner: R) -> Self:
        """Wrap an existing Rust object without re-allocating."""
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def __hash__(self) -> int:
        return hash(self._inner)

    def __eq__(self, other: object) -> bool:
        # Use isinstance against the receiver's class so subclass instances
        # of the same wrapper family (e.g. SiteLaneAddress vs LaneAddress)
        # compare equal when their underlying Rust objects are equal. Python
        # tries the reflected ``other.__eq__(self)`` if this returns
        # NotImplemented, which keeps comparison symmetric across the
        # subclass hierarchy without callers having to override.
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._inner == other._inner  # type: ignore[attr-defined]


class KirinRustWrapper(RustWrapper[R], ir.Data):
    """:class:`RustWrapper` that also satisfies Kirin's ``ir.Data`` contract.

    Use this base for Rust types that appear as IR attributes (for example
    address types). Subclasses inherit:

    * :meth:`from_inner` — runs ``__post_init__`` so the Kirin ``type``
      attribute is initialised on wrappers built from existing Rust objects.
    * :meth:`unwrap` — returns ``self._inner`` (the Rust object), which is
      the natural ``T`` for ``ir.Data[T]``.
    * :meth:`encode` — delegates to the Rust object's ``encode`` method
      (most address types implement it). Override if the Rust type does not.
    * :meth:`print_impl` — prints the encoded address as a hex literal,
      matching the legacy ``Encoder`` formatting.

    Subclasses must still define ``__init__`` to construct ``self._inner``
    and call ``self.__post_init__()``.
    """

    def __post_init__(self) -> None:
        self.type = types.PyClass(type(self))

    @classmethod
    def from_inner(cls, inner: R) -> Self:
        obj = super().from_inner(inner)
        obj.__post_init__()
        return obj

    def unwrap(self) -> R:
        return self._inner

    def encode(self) -> int:
        return self._inner.encode()  # type: ignore[attr-defined]

    def print_impl(self, printer: Printer) -> None:
        printer.plain_print(f"0x{self.encode():016x}")

    def __repr__(self) -> str:
        return f"0x{self.encode():016x}"
