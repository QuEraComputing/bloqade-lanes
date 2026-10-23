"""The ``bloqade.lanes.code_block`` dialect: register physical qubits as code blocks.

Physical-only. Add it to a kernel's dialect group and call ``register`` on each
block's qubit list; ``PhysicalNativeToPlace`` carries the membership to the
layout heuristic and the placement strategies. See
``bloqade.lanes.analysis.code_blocks``.
"""

from . import impl as impl, stmts as stmts
from ._dialect import dialect as dialect
from ._interface import register as register
from .stmts import Register as Register
