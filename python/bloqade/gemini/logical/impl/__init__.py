"""Register the Steane method tables used by Gemini logical kernels."""

from bloqade.gemini.logical.validation.clifford import impls as clifford_impls

from . import (
    measure_id as measure_id,
    measurement as measurement,
    typeinfer as typeinfer,
)

__all__ = ["clifford_impls", "measure_id", "measurement", "typeinfer"]
