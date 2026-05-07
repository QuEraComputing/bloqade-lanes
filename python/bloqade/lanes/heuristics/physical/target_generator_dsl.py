"""Starlark-hosted target generator (Plan B of #597).

Loads a ``.star`` file via the Rust ``TargetPolicyRunner`` and adapts it to
:class:`TargetGeneratorABC` so it can be injected at
``PhysicalPlacementStrategy(target_generator=...)``.

Validation (qid coverage, location-group validity, CZ-blockade pair
invariant) is performed by the Rust kernel before the candidates ever
reach Python; if the policy returns a malformed candidate, ``generate``
raises :class:`ValueError` from the underlying runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import TargetPolicyRunner
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.target_generator import (
    TargetContext,
    TargetGeneratorABC,
)


@dataclass
class TargetGeneratorDSL(TargetGeneratorABC):
    """Target generator backed by a Starlark ``.star`` policy file.

    The runner is created lazily on the first ``generate(ctx)`` call so the
    arch spec it indexes against matches the placement strategy's spec.
    Subsequent calls reuse the cached runner unless the arch spec instance
    changes (e.g. between placements over different architectures).
    """

    policy_path: str
    policy_params: Mapping[str, object] | None = None

    _runner: TargetPolicyRunner | None = field(default=None, init=False, repr=False)
    _runner_arch: ArchSpec | None = field(default=None, init=False, repr=False)

    def _ensure_runner(self, arch_spec: ArchSpec) -> TargetPolicyRunner:
        if self._runner is None or self._runner_arch is not arch_spec:
            self._runner = TargetPolicyRunner(self.policy_path, arch_spec._inner)
            self._runner_arch = arch_spec
        return self._runner

    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        runner = self._ensure_runner(ctx.arch_spec)
        # Unwrap Python LocationAddress wrappers to native Rust addresses for
        # the FFI call; rewrap the returned natives on the way back.
        placement_native = {qid: loc._inner for qid, loc in ctx.placement.items()}
        lookahead = [
            (list(controls), list(targets))
            for controls, targets in ctx.lookahead_cz_layers
        ]
        raw = runner.generate(
            placement=placement_native,
            controls=list(ctx.controls),
            targets=list(ctx.targets),
            lookahead_cz_layers=lookahead,
            cz_stage_index=ctx.cz_stage_index,
            policy_params=dict(self.policy_params) if self.policy_params else {},
        )
        return [
            {qid: LocationAddress.from_inner(loc) for qid, loc in candidate.items()}
            for candidate in raw
        ]
