"""``AtomInterpreter.get_post_processing`` must agree with
``generate_post_processing``.

The two reconstruct the same user values from the same raw measurement
array, but by different routes: ``generate_post_processing`` abstract-
interprets the *user's* kernel via ``MeasurementIDAnalysis``, while
``get_post_processing`` walks the *lowered move* kernel's analysis
output. They are documented as alternatives, so their outputs have to
match — including the leaf type and the reduce used for detectors and
observables, which is where they used to differ.
"""

import numpy as np
import pytest
from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import logical
from bloqade.gemini.post_processing import generate_post_processing
from bloqade.lanes.analysis import atom
from bloqade.lanes.analysis.atom._post_processing import constructor_function
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.transform import LogicalPipeline

# 2 logical qubits x 7 physical.
_N_RECORDS = 14


@logical.kernel(aggressive_unroll=True)
def _full():
    q = logical.qalloc_at(ilist.IList([4, 5]))
    squin.h(q[0])
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def _subset():
    q = logical.qalloc_at(ilist.IList([4, 5]))
    squin.h(q[0])
    m = logical.terminal_measure(q)
    return ilist.IList([m[1]])


@logical.kernel(aggressive_unroll=True)
def _permuted():
    q = logical.qalloc_at(ilist.IList([4, 5]))
    squin.h(q[0])
    m = logical.terminal_measure(q)
    return ilist.IList([m[1], m[0]])


def _both(user_kernel, raw):
    """Run the same shots through both implementations."""
    arch_spec = physical.get_arch_spec()
    physical_move = LogicalPipeline(transversal_rewrite=True).emit(user_kernel)
    lanes = atom.AtomInterpreter(
        physical_move.dialects, arch_spec=arch_spec
    ).get_post_processing(physical_move)

    gemini = generate_post_processing(user_kernel)
    assert gemini is not None, "the measure-id analysis should resolve this kernel"

    return list(lanes.emit_return(raw.tolist())), list(gemini(raw))


@pytest.mark.slow
@pytest.mark.parametrize(
    "user_kernel",
    [
        pytest.param(_full, id="full"),
        pytest.param(_subset, id="subset"),
        pytest.param(_permuted, id="permuted"),
    ],
)
def test_emit_return_matches_generate_post_processing(user_kernel):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, size=(4, _N_RECORDS)).astype(bool)

    lanes_out, gemini_out = _both(user_kernel, raw)
    assert lanes_out == gemini_out


@pytest.mark.slow
def test_leaf_values_are_plain_bools_from_either_route():
    """A numpy row must not leak ``np.bool_`` out of one and not the
    other; downstream equality and JSON encoding both care."""
    raw = np.ones((1, _N_RECORDS), dtype=bool)
    lanes_out, gemini_out = _both(_full, raw)

    def leaf_types(value):
        if isinstance(value, (list, tuple, ilist.IList)):
            return {t for item in value for t in leaf_types(item)}
        return {type(value)}

    assert leaf_types(lanes_out[0]) == {bool}
    assert leaf_types(gemini_out[0]) == {bool}


# ── the reduce that used to differ ───────────────────────────────


def _detector_over(*measure_results):
    return constructor_function(
        atom.DetectorResult(atom.IListResult(tuple(measure_results)))
    )


def _record(measurement_id: int):
    return atom.MeasureResult(measurement_id, 0, LocationAddress(0, 0, 0))


def test_detector_xor_matches_numpy_reduce():
    func = _detector_over(_record(0), _record(1), _record(2))
    assert func is not None
    for bits in ([True, False, False], [True, True, False], [True, True, True]):
        assert func(bits) is bool(np.logical_xor.reduce(bits, axis=0))


def test_detector_over_no_measurements_is_false_not_an_error():
    """``functools.reduce`` with no initial value raises on an empty
    sequence; ``generate_post_processing`` reduces to False, so this one
    must too."""
    func = _detector_over()
    assert func is not None
    assert func([True, False]) is False


def test_detector_result_is_a_plain_bool():
    """A numpy row is accepted at runtime even though the parameter is
    annotated ``Sequence[bool]``, and must not leak ``np.bool_``."""
    func = _detector_over(_record(0))
    assert func is not None
    assert func(np.array([True])) is True  # type: ignore[arg-type]
