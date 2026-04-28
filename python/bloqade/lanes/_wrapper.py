"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 3) for the rationale. Removed in the final cleanup stage once all
in-flight branches have rebased onto ``bloqade.lanes.bytecode._wrapper``.
"""

from bloqade.lanes.bytecode._wrapper import (
    KirinRustWrapper as KirinRustWrapper,
    RustWrapper as RustWrapper,
)
