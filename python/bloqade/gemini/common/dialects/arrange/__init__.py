"""Deprecated compatibility imports; use ``bloqade.lanes.dialects.arrange``."""

from bloqade.lanes.dialects.arrange import (
    dialect as dialect,
    move_to as move_to,
    permute as permute,
    stmts as stmts,
)

__all__ = ["dialect", "move_to", "permute", "stmts"]
