"""Compatibility re-exports for public decoder DEM utilities."""

from beliefmatching import detector_error_model_to_check_matrices
from bloqade.decoders.dem import (
    detector_error_model_matrices,
    make_layout_only_dem,
    matrix_to_dem,
)

from bloqade.gemini.decoding.mld import _select_output_observables

_make_layout_only_dem = make_layout_only_dem
_matrix_to_dem = matrix_to_dem
_compute_dem_data = detector_error_model_matrices

__all__ = [
    "_compute_dem_data",
    "_make_layout_only_dem",
    "_matrix_to_dem",
    "_select_output_observables",
    "detector_error_model_matrices",
    "detector_error_model_to_check_matrices",
    "make_layout_only_dem",
    "matrix_to_dem",
]
