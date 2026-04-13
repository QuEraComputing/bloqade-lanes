"""Tests for the RustWrapper / KirinRustWrapper base classes (#466)."""

from __future__ import annotations

from kirin import ir, types

from bloqade.lanes._wrapper import KirinRustWrapper, RustWrapper
from bloqade.lanes.bytecode._native import LocationAddress as RustLocationAddress

# ── Test fixtures: minimal subclasses backed by a real Rust type ──


class _RustWrapped(RustWrapper[RustLocationAddress]):
    """A trivial RustWrapper subclass for testing."""

    def __init__(self, zone_id: int, word_id: int, site_id: int) -> None:
        self._inner = RustLocationAddress(zone_id, word_id, site_id)


class _KirinWrapped(KirinRustWrapper[RustLocationAddress]):
    """A trivial KirinRustWrapper subclass for testing."""

    def __init__(self, zone_id: int, word_id: int, site_id: int) -> None:
        self._inner = RustLocationAddress(zone_id, word_id, site_id)
        self.__post_init__()


# ── RustWrapper ──


def test_rustwrapper_from_inner_reuses_inner_object() -> None:
    a = _RustWrapped(1, 2, 3)
    b = _RustWrapped.from_inner(a._inner)
    # Same Rust object — no re-allocation
    assert b._inner is a._inner


def test_rustwrapper_eq_and_hash_delegate_to_inner() -> None:
    a = _RustWrapped(1, 2, 3)
    b = _RustWrapped(1, 2, 3)  # different Rust object, same field values
    assert a == b
    assert hash(a) == hash(b)


def test_rustwrapper_eq_distinguishes_distinct_field_values() -> None:
    a = _RustWrapped(1, 2, 3)
    c = _RustWrapped(1, 2, 4)
    assert a != c
    assert hash(a) != hash(c)


def test_rustwrapper_eq_returns_notimplemented_for_other_type() -> None:
    a = _RustWrapped(1, 2, 3)
    # __eq__ returns NotImplemented for foreign types; Python then returns False
    assert (a == "not a wrapper") is False
    assert (a == object()) is False


def test_rustwrapper_eq_distinguishes_unrelated_subclasses() -> None:
    """Sibling subclasses (no inheritance between them) never compare equal."""

    class _Other(RustWrapper[RustLocationAddress]):
        def __init__(self, z: int, w: int, s: int) -> None:
            self._inner = RustLocationAddress(z, w, s)

    a = _RustWrapped(1, 2, 3)
    b = _Other(1, 2, 3)
    # Different wrapper families; both __eq__ calls return NotImplemented.
    assert (a == b) is False
    assert (b == a) is False


def test_rustwrapper_eq_allows_subclass_against_base() -> None:
    """A subclass instance compares equal to a base-class instance with
    the same underlying Rust object — matches the legacy LaneAddress
    semantic where ``WordLaneAddress(0, 0, 0) == LaneAddress(WORD, ...)``.
    """

    class _Sub(_RustWrapped):
        pass

    base = _RustWrapped(1, 2, 3)
    sub = _Sub(1, 2, 3)
    assert base == sub
    assert sub == base
    assert hash(base) == hash(sub)


# ── KirinRustWrapper ──


def test_kirinwrapper_is_an_ir_data() -> None:
    x = _KirinWrapped(1, 2, 3)
    assert isinstance(x, ir.Data)


def test_kirinwrapper_post_init_sets_kirin_type() -> None:
    x = _KirinWrapped(1, 2, 3)
    assert isinstance(x.type, types.PyClass)
    assert x.type.typ is _KirinWrapped


def test_kirinwrapper_from_inner_runs_post_init() -> None:
    """Wrappers built via from_inner must still satisfy the ir.Data contract."""
    rust = RustLocationAddress(1, 2, 3)
    x = _KirinWrapped.from_inner(rust)
    assert x._inner is rust
    assert hasattr(x, "type"), "from_inner must initialise the Kirin type"
    assert isinstance(x.type, types.PyClass)


def test_kirinwrapper_unwrap_returns_rust_object() -> None:
    """unwrap() returns the Rust object, matching ir.Data[T] semantics."""
    x = _KirinWrapped(1, 2, 3)
    assert x.unwrap() is x._inner
    assert isinstance(x.unwrap(), RustLocationAddress)


def test_kirinwrapper_encode_delegates_to_inner() -> None:
    x = _KirinWrapped(1, 2, 3)
    assert x.encode() == x._inner.encode()


def test_kirinwrapper_repr_is_hex_encoded() -> None:
    x = _KirinWrapped(1, 2, 3)
    expected = f"0x{x.encode():016x}"
    assert repr(x) == expected


def test_kirinwrapper_eq_and_hash_inherited() -> None:
    """KirinRustWrapper still gets RustWrapper's eq/hash semantics."""
    a = _KirinWrapped(1, 2, 3)
    b = _KirinWrapped.from_inner(RustLocationAddress(1, 2, 3))
    assert a == b
    assert hash(a) == hash(b)
