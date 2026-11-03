"""Deprecated module for importing TomographyResult; please import from bloqade-circuit instead."""

import warnings

from bloqade.analysis.tomography import TomographyResult  # noqa: F401

warnings.warn(
    'The "bloqade.gemini.decoding.tomography.TomographyResult" import is deprecated; please import from "bloqade.analysis.tomography.TomographyResult" instead.',
    FutureWarning,
    stacklevel=2,
)
