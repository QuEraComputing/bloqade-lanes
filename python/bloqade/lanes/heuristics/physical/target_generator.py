"""Target-generation plugin interface for ``PhysicalPlacementStrategy``.

``PhysicalPlacementStrategy`` asks a :class:`TargetGeneratorABC` for an
ordered list of candidate target placements before each CZ stage. The
strategy always appends :class:`DefaultTargetGenerator`'s output as a
guaranteed last-resort fallback, so plugins may return ``[]`` to defer
entirely to the default.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.layout import LocationAddress


@dataclass(frozen=True)
class TargetContext:
    """Signals passed to a TargetGenerator.

    Composes ConcreteState to avoid duplicating lattice state fields.
    """

    arch_spec: layout.ArchSpec
    state: ConcreteState
    controls: tuple[int, ...]
    targets: tuple[int, ...]
    lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    cz_stage_index: int

    @property
    def placement(self) -> dict[int, LocationAddress]:
        return dict(enumerate(self.state.layout))


class TargetGeneratorABC(abc.ABC):
    """Plugin interface for choosing the target configuration of a CZ stage.

    Implementations return an *ordered* list of candidate target
    placements. The strategy framework appends the default candidate
    (``DefaultTargetGenerator``) as a guaranteed last-resort, so a plugin
    may return ``[]`` to defer entirely to the default.
    """

    @abc.abstractmethod
    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]: ...


@dataclass(frozen=True)
class DefaultTargetGenerator(TargetGeneratorABC):
    """Default rule: control qubit moves to the CZ partner of the target's location."""

    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        target = dict(ctx.placement)
        for control_qid, target_qid in zip(ctx.controls, ctx.targets):
            target_loc = target[target_qid]
            partner = ctx.arch_spec.get_cz_partner(target_loc)
            assert partner is not None, f"No CZ blockade partner for {target_loc}"
            target[control_qid] = partner
        return [target]


TargetGeneratorCallable = Callable[[TargetContext], list[dict[int, LocationAddress]]]


@dataclass(frozen=True)
class _CallableTargetGenerator(TargetGeneratorABC):
    """Private adapter that lifts a bare callable to TargetGeneratorABC."""

    fn: TargetGeneratorCallable

    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        return self.fn(ctx)


def _coerce_target_generator(
    value: TargetGeneratorABC | TargetGeneratorCallable | None,
) -> TargetGeneratorABC | None:
    """Normalize the public union down to TargetGeneratorABC | None."""
    if value is None or isinstance(value, TargetGeneratorABC):
        return value
    return _CallableTargetGenerator(value)


def _validate_candidate(
    ctx: TargetContext,
    candidate: dict[int, LocationAddress],
) -> None:
    """Raise ValueError if the candidate is not a legal CZ target.

    Checks:
    1. Every qid from ``ctx.placement`` appears in ``candidate``; no
       unexpected extra qids are present.
    2. Every location value is recognized by ``ctx.arch_spec`` (via
       ``check_location_group``). Group-level errors such as duplicate
       locations are caught here.
    3. Each ``(control_qid, target_qid)`` pair is CZ-blockade-partnered
       in either direction (matching the convention at
       ``python/bloqade/lanes/analysis/placement/lattice.py:134-135``).
    """
    placement = ctx.placement
    missing = set(placement.keys()) - set(candidate.keys())
    extra = set(candidate.keys()) - set(placement.keys())
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unexpected {sorted(extra)}")
        raise ValueError(f"target-generator candidate qubits: {'; '.join(parts)}")
    # Run the Rust-backed validator on the full candidate so group-level
    # errors (e.g. duplicate locations) are caught, then attribute per-qid
    # where possible for a helpful error message.
    group_errors = list(ctx.arch_spec.check_location_group(list(candidate.values())))
    if group_errors:
        per_qid_bad = [
            f"qid={qid} @ {loc}"
            for qid, loc in candidate.items()
            if ctx.arch_spec.check_location_group([loc])
        ]
        detail = per_qid_bad if per_qid_bad else [str(e) for e in group_errors]
        raise ValueError(f"target-generator candidate has invalid locations: {detail}")
    for control_qid, target_qid in zip(ctx.controls, ctx.targets):
        c_loc = candidate[control_qid]
        t_loc = candidate[target_qid]
        if (
            ctx.arch_spec.get_cz_partner(c_loc) != t_loc
            and ctx.arch_spec.get_cz_partner(t_loc) != c_loc
        ):
            raise ValueError(
                f"target-generator candidate CZ pair "
                f"(control={control_qid}@{c_loc}, target={target_qid}@{t_loc}) "
                f"is not blockade-partnered"
            )
