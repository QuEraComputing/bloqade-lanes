from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pack_boolean_array(arr: np.ndarray) -> np.ndarray:
    """Pack each row of a bit array into an unsigned integer.

    Args:
        arr: One- or two-dimensional array of 0/1 values. Bits are interpreted
            little-endian along the final axis.

    Returns:
        One packed unsigned integer per input row.
    """

    arr = np.asarray(arr, dtype=np.uint64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.sum(arr << np.arange(arr.shape[1], dtype=np.uint64), axis=1)


def packed_bits_to_int(bits: np.ndarray | Sequence[bool] | Sequence[int]) -> int:
    """Pack a one-dimensional bit sequence into a Python integer.

    Args:
        bits: Sequence of boolean or 0/1 values interpreted little-endian.

    Returns:
        The packed integer value.
    """

    return int(pack_boolean_array(np.asarray(bits, dtype=np.uint8))[0])


def unpack_packed_bits(packed: int, length: int) -> np.ndarray:
    """Unpack an integer into a little-endian ``uint8`` bit vector.

    Args:
        packed: Integer value to unpack.
        length: Number of bits to return.

    Returns:
        A length-``length`` vector of 0/1 values.
    """

    return ((int(packed) >> np.arange(length, dtype=np.uint64)) & 1).astype(np.uint8)


def _packed_pattern_targets(
    targets: np.ndarray,
) -> set[int]:
    """Pack a target syndrome matrix into an integer lookup set."""

    return {
        int(x) for x in pack_boolean_array(np.asarray(targets, dtype=np.uint8)).tolist()
    }
