"""Compatibility re-exports for Gemini syndrome-layout helpers."""

from bloqade.gemini.decoding.layout import (
    DEFAULT_SYNDROME_LAYOUT,
    SyndromeLayout,
    split_factory_bits,
)

__all__ = [
    "DEFAULT_SYNDROME_LAYOUT",
    "SyndromeLayout",
    "split_factory_bits",
]
