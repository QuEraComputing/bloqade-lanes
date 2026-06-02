from __future__ import annotations

from bloqade.analysis.tomography import DEFAULT_TARGET_BLOCH

DEFAULT_BASIS_LABELS = ("X", "Y", "Z")
DEFAULT_IDEAL_FACTORY_ACCEPTANCE = 1.0 / 6.0

__all__ = [
    "DEFAULT_BASIS_LABELS",
    "DEFAULT_IDEAL_FACTORY_ACCEPTANCE",
    "DEFAULT_TARGET_BLOCH",
]
