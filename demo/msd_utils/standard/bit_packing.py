from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pack_boolean_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.uint64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.sum(arr << np.arange(arr.shape[1], dtype=np.uint64), axis=1)


def packed_bits_to_int(bits: np.ndarray | Sequence[bool] | Sequence[int]) -> int:
    return int(pack_boolean_array(np.asarray(bits, dtype=np.uint8))[0])


def unpack_packed_bits(packed: int, length: int) -> np.ndarray:
    return ((int(packed) >> np.arange(length, dtype=np.uint64)) & 1).astype(np.uint8)


def _packed_pattern_targets(
    targets: np.ndarray,
) -> set[int]:
    return {
        int(x) for x in pack_boolean_array(np.asarray(targets, dtype=np.uint8)).tolist()
    }
