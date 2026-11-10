"""Check externally supplied interpreter implementations before running analysis."""

from collections.abc import Iterable

from kirin import ir


def require_method_tables(
    method: ir.Method, requirements: Iterable[tuple[ir.Dialect, str]]
) -> None:
    """Reject missing tables for dialects actually present in this method's IR.

    Checking statements, rather than the whole dialect group, keeps validation
    of lowered Move programs independent of unused frontend dialects. Gemini's
    kernel groups register its tables; other frontends may provide their own.
    """
    present = {stmt.dialect for stmt in method.code.walk()}
    missing = [
        f"{dialect.name}:{key}"
        for dialect, key in requirements
        if dialect in present and key not in dialect.interps
    ]
    if missing:
        raise ValueError(
            "Missing compiler method tables: "
            + ", ".join(missing)
            + ". Register the frontend's method tables before compiling or "
            "validating this kernel. Gemini callers should import "
            "bloqade.gemini.logical or bloqade.gemini.physical first."
        )
