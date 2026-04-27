"""Shot-remapping helper.

A *shot* coming back from hardware is a full Zone-0 bitstring: one
bit per ``LocationAddress`` in ``arch_spec.yield_zone_locations(
ZoneAddress(0))``, including bits for sites that no atom is ever
moved into. Downstream post-processing (detector / observable
synthesis, user-level result reconstruction) operates on a *per-
measurement* array — typically shape ``(n_shots, n_measurements)``
— that's been projected out of the full bitstring at exactly the
sites the program actually measures.

This module provides the bridge: given the analysis output for a
``terminal_measure`` (or equivalent) ``IListResult[IListResult[
MeasureResult]]`` value, plus the architecture spec, build the
``logical_mapping[i]`` index table the original #98 spec called for.

See: issue #563.
"""

from __future__ import annotations

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import ZoneAddress

from .lattice import IListResult, MeasureResult, MoveExecution


def get_shot_remapping(
    return_value: MoveExecution,
    arch_spec: ArchSpec,
) -> list[list[int]] | None:
    """Project an analysis ``IListResult[IListResult[MeasureResult]]``
    value onto the corresponding Zone-0 bitstring indices.

    Args:
        return_value: lattice value for the SSA result of a
            ``terminal_measure`` (or any value with the nested-IList
            shape produced by lowering a logical-qubit measurement
            through the atom-analysis chain). The outer ``IListResult``
            indexes logical qubits; each inner ``IListResult`` indexes
            the physical qubits making up that logical block.
        arch_spec: architecture spec; ``arch_spec.yield_zone_locations(
            ZoneAddress(0))`` defines the canonical Zone-0 bitstring
            layout that hardware shots are reported against.

    Returns:
        ``list[list[int]]`` where ``result[i][j]`` is the position in
        the Zone-0 bitstring corresponding to the ``j``-th physical
        qubit of the ``i``-th logical qubit, **or** ``None`` if the
        mapping cannot be derived. Reasons the mapping may fail:

        - ``return_value`` (or any nested element) does not have the
          expected ``IListResult[IListResult[MeasureResult]]`` shape
          — most often a sign that the analysis didn't refine the
          lattice value past ``Bottom`` / ``Unknown``.
        - A ``MeasureResult.location_address`` resolves outside
          ``arch_spec``'s Zone-0 iteration — i.e. the analysis and
          arch spec disagree about hardware layout.

        Callers are expected to surface a meaningful diagnostic when
        ``None`` is returned; this function does not raise on either
        condition.
    """
    zone0 = ZoneAddress(0)

    if not isinstance(return_value, IListResult):
        return None

    remapping: list[list[int]] = []
    for logical in return_value.data:
        if not isinstance(logical, IListResult):
            return None
        per_qubit: list[int] = []
        for physical in logical.data:
            if not isinstance(physical, MeasureResult):
                return None
            # ``ArchSpec.get_zone_index`` is O(1) via the Rust backend
            # and returns ``None`` for addresses outside Zone-0 — which
            # propagates as the "give up" signal here.
            idx = arch_spec.get_zone_index(physical.location_address, zone0)
            if idx is None:
                return None
            per_qubit.append(idx)
        remapping.append(per_qubit)
    return remapping
