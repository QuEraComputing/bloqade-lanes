"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 8) for the rationale. Removed in the final cleanup stage once all
in-flight branches have rebased onto ``bloqade.lanes.arch.build.word_factory``.
"""

from bloqade.lanes.arch.build.word_factory import (
    WordGrid as WordGrid,
    create_zone_words as create_zone_words,
)
