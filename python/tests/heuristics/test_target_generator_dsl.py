"""Tests for the Starlark-hosted TargetGeneratorDSL adapter (Plan B of #597)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.target_generator import (
    DefaultTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
)
from bloqade.lanes.heuristics.physical.target_generator_dsl import TargetGeneratorDSL

DEFAULT_POLICY = """
def generate(ctx, lib):
    target = {}
    for q in ctx.placement.qubits():
        target[q] = ctx.placement.get(q)
    for i in range(len(ctx.controls)):
        c = ctx.controls[i]
        t = ctx.targets[i]
        target[c] = lib.cz_partner(target[t])
    return [target]
"""

EMPTY_POLICY = """
def generate(ctx, lib):
    return []
"""


def _write_policy(src: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".star", delete=False)
    f.write(src)
    f.flush()
    f.close()
    return f.name


def _make_state() -> ConcreteState:
    return ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(1, 0)),
        move_count=(0, 0),
    )


def _make_ctx() -> TargetContext:
    return TargetContext(
        arch_spec=logical.get_arch_spec(),
        state=_make_state(),
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )


def test_dsl_subclasses_target_generator_abc():
    assert issubclass(TargetGeneratorDSL, TargetGeneratorABC)


def test_dsl_default_policy_matches_default_generator():
    path = _write_policy(DEFAULT_POLICY)
    try:
        ctx = _make_ctx()
        dsl_out = TargetGeneratorDSL(policy_path=path).generate(ctx)
        ref_out = DefaultTargetGenerator().generate(ctx)
        assert len(dsl_out) == 1
        assert dsl_out[0] == ref_out[0]
    finally:
        Path(path).unlink(missing_ok=True)


def test_dsl_returns_python_location_address_wrappers():
    path = _write_policy(DEFAULT_POLICY)
    try:
        out = TargetGeneratorDSL(policy_path=path).generate(_make_ctx())
        assert len(out) == 1
        for loc in out[0].values():
            assert isinstance(
                loc, LocationAddress
            ), f"expected LocationAddress wrapper, got {type(loc).__name__}"
    finally:
        Path(path).unlink(missing_ok=True)


def test_dsl_empty_policy_returns_no_candidates():
    path = _write_policy(EMPTY_POLICY)
    try:
        out = TargetGeneratorDSL(policy_path=path).generate(_make_ctx())
        assert out == []
    finally:
        Path(path).unlink(missing_ok=True)


def test_dsl_invalid_candidate_raises_value_error():
    # Place both qubits at the same word so the no-op policy below leaves
    # them as non-CZ-partners, which must trip the kernel's validator.
    bad_state = ConcreteState(
        occupied=frozenset(),
        layout=(LocationAddress(0, 0), LocationAddress(0, 1)),
        move_count=(0, 0),
    )
    bad_ctx = TargetContext(
        arch_spec=logical.get_arch_spec(),
        state=bad_state,
        controls=(0,),
        targets=(1,),
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )
    bad = """
def generate(ctx, lib):
    bad = {}
    for q in ctx.placement.qubits():
        bad[q] = ctx.placement.get(q)
    return [bad]
"""
    path = _write_policy(bad)
    try:
        with pytest.raises(ValueError):
            TargetGeneratorDSL(policy_path=path).generate(bad_ctx)
    finally:
        Path(path).unlink(missing_ok=True)


def test_dsl_caches_runner_across_calls():
    path = _write_policy(DEFAULT_POLICY)
    try:
        gen = TargetGeneratorDSL(policy_path=path)
        ctx = _make_ctx()
        gen.generate(ctx)
        first_runner = gen._runner
        gen.generate(ctx)
        assert gen._runner is first_runner, "runner should be cached"
    finally:
        Path(path).unlink(missing_ok=True)


def test_reference_default_target_policy_matches_default_generator():
    repo_root = Path(__file__).resolve().parents[3]
    policy = repo_root / "policies" / "reference" / "default_target.star"
    assert policy.exists(), f"reference policy missing at {policy}"
    ctx = _make_ctx()
    out = TargetGeneratorDSL(policy_path=str(policy)).generate(ctx)
    ref = DefaultTargetGenerator().generate(ctx)
    assert len(out) == 1
    assert out[0] == ref[0]


def test_strategy_accepts_dsl_generator_end_to_end():
    """Wire TargetGeneratorDSL into PhysicalPlacementStrategy and verify
    the strategy reaches the same candidate set as the in-tree default."""
    from bloqade.lanes.heuristics.physical.movement import PhysicalPlacementStrategy

    repo_root = Path(__file__).resolve().parents[3]
    policy = str(repo_root / "policies" / "reference" / "default_target.star")
    arch_spec = logical.get_arch_spec()
    strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        target_generator=TargetGeneratorDSL(policy_path=policy),
    )
    ctx = _make_ctx()
    candidates = strategy._build_candidates(ctx)
    # Reference policy mirrors DefaultTargetGenerator → after dedup, one
    # candidate survives (the same one DefaultTargetGenerator would emit).
    assert len(candidates) == 1
    assert candidates[0] == DefaultTargetGenerator().generate(ctx)[0]
