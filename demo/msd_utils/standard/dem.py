"""Compatibility re-exports for Gemini decoding DEM helpers."""

from bloqade.gemini.decoding.dem import (
    DetectorErrorModelTask,
    _compute_dem_data,
    _make_layout_only_dem,
    _matrix_to_dem,
    detector_error_model_matrices,
    detector_error_model_to_check_matrices,
    make_layout_only_dem,
    matrix_to_dem,
    sub_detector_error_model,
)

__all__ = [
    "DetectorErrorModelTask",
    "_compute_dem_data",
    "_make_layout_only_dem",
    "_matrix_to_dem",
    "detector_error_model_matrices",
    "detector_error_model_to_check_matrices",
    "make_layout_only_dem",
    "matrix_to_dem",
    "sub_detector_error_model",
]
