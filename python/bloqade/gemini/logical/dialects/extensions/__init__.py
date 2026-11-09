"""Experimental logical-level extensions.

Statements here are part of the `@gemini.logical.kernel` dialect group like any
other logical operation, but they are *not* logical operations in the
error-corrected sense: they act on the physical qubits underneath a logical
qubit and take the state out of the logical subspace. They are separated out so
that the distinction is visible in the import path -- see
`bloqade.gemini.logical.extensions` for the user-facing wrappers.
"""

from . import stmts as stmts
from ._dialect import dialect as dialect
from ._interface import star_rz as star_rz
