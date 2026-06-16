"""Bit-packing utilities for detector and observable shots."""

from __future__ import annotations

from collections.abc import Sequence

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


def unpack_boolean_array(arr: np.ndarray, data_len: int) -> npt.NDArray[np.bool_]:
    """Unpack integer labels into a little-endian boolean bit matrix."""

    packed = np.asarray(arr, dtype=np.uint64).reshape(-1, 1)
    if data_len < 0:
        raise ValueError("data_len must be non-negative.")
    if data_len == 0:
        return np.zeros((packed.shape[0], 0), dtype=np.bool_)
    shifts = np.arange(data_len, dtype=np.uint64).reshape(1, -1)
    return ((packed >> shifts) & 1).astype(np.bool_)


def packed_bits_to_int(bits: np.ndarray | Sequence[bool] | Sequence[int]) -> int:
    """Pack a one-dimensional bit sequence into a Python integer."""

    return int(pack_boolean_array(np.asarray(bits, dtype=np.uint8))[0])


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


def det_obs_shots_to_counts(
    det_shots: np.ndarray,
    obs_shots: np.ndarray,
) -> npt.NDArray[np.int64]:
    """Convert detector and observable shots into a combined count table."""

    return shots_to_counts(np.concatenate([det_shots, obs_shots], axis=1))


def packed_pattern_targets(targets: np.ndarray) -> set[int]:
    """Pack a target syndrome matrix into an integer lookup set."""

    return {int(x) for x in pack_boolean_array(np.asarray(targets, dtype=np.uint8))}


_packed_pattern_targets = packed_pattern_targets


__all__ = [
    "_packed_pattern_targets",
    "det_obs_shots_to_counts",
    "pack_boolean_array",
    "packed_bits_to_int",
    "packed_pattern_targets",
    "shots_to_counts",
    "unpack_boolean_array",
    "unpack_packed_bits",
]
