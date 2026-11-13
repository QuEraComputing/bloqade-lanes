"""Compiler fixtures for code-block registration and deferred slot allocation."""

from bloqade import squin
from bloqade.lanes.dialects import arch, code_block

kernel = squin.kernel.add(arch).add(code_block)
