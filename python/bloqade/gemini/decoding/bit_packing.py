"""Bit-packing utilities for detector and observable shots."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def pack_boolean_array(arr: np.ndarray) -> npt.NDArray[np.uint64]:
    """Pack each row of a bit array into an unsigned integer.

    Bits are interpreted little-endian along the final axis, so the first
    column contributes the least-significant bit.
    """

    bits = np.asarray(arr, dtype=np.uint64)
    if bits.ndim == 1:
        bits = bits.reshape(1, -1)
    if bits.ndim != 2:
        raise ValueError("arr must be a 1D or 2D bit array.")
    if bits.shape[1] > 64:
        raise ValueError("Cannot pack more than 64 bits into uint64 values.")
    shifts = np.arange(bits.shape[1], dtype=np.uint64)
    return np.sum(bits << shifts, axis=1).astype(np.uint64, copy=False)


def unpack_packed_bits(packed: int, length: int) -> npt.NDArray[np.uint8]:
    """Unpack a Python integer into a little-endian ``uint8`` bit vector."""

    if length < 0:
        raise ValueError("length must be non-negative.")
    if length == 0:
        return np.zeros(0, dtype=np.uint8)
    shifts = np.arange(length, dtype=np.uint64)
    return ((np.uint64(packed) >> shifts) & 1).astype(np.uint8)


def shots_to_counts(shots: np.ndarray) -> npt.NDArray[np.int64]:
    """Convert a boolean shot matrix into a dense little-endian count table."""

    shot_bits = np.asarray(shots, dtype=np.uint8)
    if shot_bits.ndim != 2:
        raise ValueError("shots must be a 2D bit array.")
    packed_shots = pack_boolean_array(shot_bits)
    return np.bincount(
        packed_shots.astype(np.int64, copy=False),
        minlength=1 << shot_bits.shape[1],
    )


def packed_pattern_targets(targets: np.ndarray) -> set[int]:
    """Pack a target syndrome matrix into an integer lookup set."""

    return {int(x) for x in pack_boolean_array(np.asarray(targets, dtype=np.uint8))}


__all__ = [
    "pack_boolean_array",
    "packed_pattern_targets",
    "shots_to_counts",
    "unpack_packed_bits",
]
