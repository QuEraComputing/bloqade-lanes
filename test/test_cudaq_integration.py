from __future__ import annotations

import pytest
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.logical.dialects.operations.stmts import (
    TerminalLogicalMeasurement,
)
from kirin.dialects import func

from bloqade import qubit, squin
from bloqade.lanes.logical_mvp import (
    _find_qubit_ssas,
    _find_return_stmt,
    append_measurements_and_annotations,
)

DETS = [[1, 0], [0, 1]]  # 2 measurements, 2 detectors
OBS = [[1], [1]]  # 2 measurements, 1 observable


def _make_kernel(num_qubits: int, *, with_measure: bool = False):
    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(num_qubits)
        squin.h(reg[0])
        if with_measure:
            gemini_logical.terminal_measure(reg)

    return kernel


def _count_stmts(mt, stmt_type) -> int:
    return sum(1 for s in mt.callable_region.walk() if isinstance(s, stmt_type))


@pytest.mark.parametrize("n", [1, 3])
def test_find_qubit_ssas(n: int):
    assert len(_find_qubit_ssas(_make_kernel(n))) == n


def test_find_qubit_ssas_no_qubits():
    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        return 42

    assert len(_find_qubit_ssas(kernel)) == 0


def test_find_return_stmt():
    mt = _make_kernel(1)
    ret = _find_return_stmt(mt)
    assert isinstance(ret, func.Return)
    assert ret is mt.callable_region.blocks[0].last_stmt


def test_raises_when_both_matrices_none():
    with pytest.raises(ValueError, match="At least one"):
        append_measurements_and_annotations(_make_kernel(1), None, None)


def test_raises_when_no_qubits():
    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        return 42

    with pytest.raises(ValueError, match="No qubit allocations"):
        append_measurements_and_annotations(kernel, DETS, OBS)


@pytest.mark.parametrize(
    "with_measure", [False, True], ids=["no_terminal", "has_terminal"]
)
def test_terminal_measurement_count(with_measure: bool):
    mt = _make_kernel(1, with_measure=with_measure)
    append_measurements_and_annotations(mt, DETS, OBS)
    assert _count_stmts(mt, TerminalLogicalMeasurement) == 1


@pytest.mark.parametrize(
    "dets, obs, expected_dets, expected_obs",
    [
        (DETS, OBS, 2, 1),
        (DETS, None, 2, 0),
        (None, OBS, 0, 1),
    ],
    ids=["both", "dets_only", "obs_only"],
)
def test_inserted_annotation_counts(dets, obs, expected_dets, expected_obs):
    mt = _make_kernel(1)
    append_measurements_and_annotations(mt, dets, obs)
    assert _count_stmts(mt, SetDetector) == expected_dets
    assert _count_stmts(mt, SetObservable) == expected_obs


def test_rewrites_return_when_no_existing_terminal():
    mt = _make_kernel(1)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)
    append_measurements_and_annotations(mt, DETS, OBS)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)


def test_rewrites_return_when_terminal_exists():
    mt = _make_kernel(1, with_measure=True)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)
    append_measurements_and_annotations(mt, DETS, OBS)
    assert isinstance(_find_return_stmt(mt).value.owner, func.ConstantNone)
