"""Compatibility re-exports for public decoder bit-packing utilities."""

from bloqade.decoders.bit_packing import (
    det_obs_shots_to_counts,
    pack_boolean_array,
    packed_bits_to_int,
    packed_pattern_targets,
    shots_to_counts,
    unpack_boolean_array,
    unpack_packed_bits,
)

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
