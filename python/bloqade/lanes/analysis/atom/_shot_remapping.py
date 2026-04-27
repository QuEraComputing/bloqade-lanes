"""Shot-remapping helper.

A *shot* coming back from hardware is a full Zone-0 bitstring: one
bit per ``LocationAddress`` in ``arch_spec.yield_zone_locations(
ZoneAddress(0))``, including bits for sites that no atom is ever
moved into. Downstream post-processing (detector / observable
synthesis, user-level result reconstruction) operates on a *per-
measurement* flat array — typically shape ``(n_shots, n_measurements)``
— that's been projected out of the full bitstring at exactly the
sites the program actually measures, in the order the post-processing
callable expects.

This module provides the bridge: given the analysis output for a
``terminal_measure`` (or equivalent) ``IListResult[IListResult[
MeasureResult]]`` value, plus the architecture spec, produce a flat
list of Zone-0 bitstring indices whose order matches the per-
measurement array consumed by ``generate_post_processing``.

See: issue #563.
"""

from __future__ import annotations

from dataclasses import dataclass

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import LocationAddress, ZoneAddress

from .lattice import IListResult, MeasureResult, MoveExecution


@dataclass(frozen=True)
class ShotRemappingDiagnostic:
    """Compiler-developer-facing diagnostic emitted when
    ``get_shot_remapping`` cannot derive a Zone-0 index list.

    A failure here indicates an analysis or pipeline regression
    rather than a user error — the user supplied a kernel, the
    compiler service lowered it, and somewhere along the way the
    analysis output drifted away from the expected
    ``IListResult[IListResult[MeasureResult]]`` shape (or pointed
    to a hardware location the architecture doesn't know about).
    The fields below carry enough context for a compiler developer
    to find the offending pass.

    Attributes:
        message: human-readable description with the failure path
            baked in (e.g. ``"logical[2].physical[5]: …"``).
        offending_value: the lattice value or address that triggered
            the failure.
    """

    message: str
    offending_value: MoveExecution | LocationAddress


@dataclass(frozen=True)
class ShotMappingResult:
    """Result of computing the Zone-0 bitstring index list for a
    move kernel's ``terminal_measure`` SSA value.

    On success ``mapping`` holds the flat list of indices and
    ``diagnostic`` is ``None``. On failure ``mapping`` is ``None``
    and ``diagnostic`` carries the debug context.
    """

    mapping: list[int] | None
    diagnostic: ShotRemappingDiagnostic | None = None

    @property
    def ok(self) -> bool:
        """``True`` iff the mapping was derived successfully."""
        return self.mapping is not None

    def get(self) -> list[int]:
        """Return the mapping or raise ``RuntimeError`` (carrying the
        diagnostic) if the computation failed."""
        if self.mapping is None:
            raise RuntimeError(
                f"ShotMappingResult: mapping unavailable; "
                f"diagnostic: {self.diagnostic}"
            )
        return self.mapping


def get_shot_remapping(
    return_value: MoveExecution,
    arch_spec: ArchSpec,
) -> ShotMappingResult:
    """Project an analysis ``IListResult[IListResult[MeasureResult]]``
    value onto a flat list of Zone-0 bitstring indices.

    Args:
        return_value: lattice value for the SSA result of a
            ``terminal_measure`` (or any value with the nested-IList
            shape produced by lowering a logical-qubit measurement
            through the atom-analysis chain). The outer ``IListResult``
            indexes logical qubits; each inner ``IListResult`` indexes
            the physical qubits making up that logical block. The
            nested structure is walked in row-major order; the result
            is flat.
        arch_spec: architecture spec; ``arch_spec.yield_zone_locations(
            ZoneAddress(0))`` defines the canonical Zone-0 bitstring
            layout that hardware shots are reported against.

    Returns:
        ``ShotMappingResult`` whose ``mapping`` is the flat list of
        Zone-0 indices in row-major order on success, or ``None`` with
        a populated ``diagnostic`` on failure. Failure modes:

        - ``return_value`` (or any nested element) does not have the
          expected ``IListResult[IListResult[MeasureResult]]`` shape
          — most often a sign that the analysis didn't refine the
          lattice value past ``Bottom`` / ``Unknown``.
        - A ``MeasureResult.location_address`` resolves outside
          ``arch_spec``'s Zone-0 iteration — i.e. the analysis and
          arch spec disagree about hardware layout.

        The diagnostic is aimed at the compiler service / compiler
        developers, not end users; failures here indicate pipeline
        regressions rather than malformed kernels.
    """
    zone0 = ZoneAddress(0)

    if not isinstance(return_value, IListResult):
        return ShotMappingResult(
            mapping=None,
            diagnostic=ShotRemappingDiagnostic(
                message=(
                    "outer return value did not refine to IListResult; "
                    f"got {type(return_value).__name__}"
                ),
                offending_value=return_value,
            ),
        )

    remapping: list[int] = []
    for i, logical in enumerate(return_value.data):
        if not isinstance(logical, IListResult):
            return ShotMappingResult(
                mapping=None,
                diagnostic=ShotRemappingDiagnostic(
                    message=(
                        f"logical[{i}] did not refine to IListResult; "
                        f"got {type(logical).__name__}"
                    ),
                    offending_value=logical,
                ),
            )
        for j, physical in enumerate(logical.data):
            if not isinstance(physical, MeasureResult):
                return ShotMappingResult(
                    mapping=None,
                    diagnostic=ShotRemappingDiagnostic(
                        message=(
                            f"logical[{i}].physical[{j}] did not refine "
                            f"to MeasureResult; got {type(physical).__name__}"
                        ),
                        offending_value=physical,
                    ),
                )
            # ``ArchSpec.get_zone_index`` is O(1) via the Rust backend
            # and returns ``None`` for addresses outside Zone-0.
            idx = arch_spec.get_zone_index(physical.location_address, zone0)
            if idx is None:
                return ShotMappingResult(
                    mapping=None,
                    diagnostic=ShotRemappingDiagnostic(
                        message=(
                            f"logical[{i}].physical[{j}]: "
                            f"location_address {physical.location_address} "
                            "is not in Zone-0 of the arch spec"
                        ),
                        offending_value=physical.location_address,
                    ),
                )
            remapping.append(idx)
    return ShotMappingResult(mapping=remapping)
