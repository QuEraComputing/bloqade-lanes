"""Entropy-guided search tree visualizer.

Public API:
    - :func:`build_entropy_trace` -- run the compilation pipeline and return an
      :class:`EntropyTraceBundle` wrapping the Rust solver's entropy trace.
    - :class:`TreeFrameState`, :class:`TreeStateReducer` -- frame-level state
      derived from a trace for rendering.
    - :func:`run` -- CLI entry point (also accessible via
      ``python -m bloqade.lanes.visualize.entropy_tree``).
"""

from bloqade.lanes.visualize.entropy_tree.cli import run
from bloqade.lanes.visualize.entropy_tree.state import (
    TreeFrameState,
    TreeStateReducer,
)
from bloqade.lanes.visualize.entropy_tree.tracer import (
    EntropyTraceBundle,
    build_entropy_trace,
)

__all__ = [
    "EntropyTraceBundle",
    "TreeFrameState",
    "TreeStateReducer",
    "build_entropy_trace",
    "run",
]
