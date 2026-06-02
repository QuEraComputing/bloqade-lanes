"""Compatibility re-exports for table decoder classes."""

from bloqade.decoders import SparseTableDecoder, TableDecoder

from bloqade.gemini.decoding.types import TableDecoderClass

__all__ = ["SparseTableDecoder", "TableDecoder", "TableDecoderClass"]
