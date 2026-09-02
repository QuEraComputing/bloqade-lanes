import bloqade.squin as squin
import numpy as np
from kirin.dialects import ilist
from kirin.interp.exceptions import InterpreterError

import bloqade.gemini as gemini
from bloqade.gemini import post_processing as post_processing_module
from bloqade.gemini.logical import kernel
from bloqade.gemini.post_processing import (
    build_post_processing,
    generate_post_processing,
)
from bloqade.lanes.analysis import atom


def test_none():

    @kernel
    def main():
        return

    post_processing = build_post_processing(main)
    raw_results: list[list[bool]] = [[], []]

    assert list(post_processing.emit_return(raw_results)) == [None, None]
    assert list(post_processing.emit_detectors(raw_results)) == [[], []]
    assert list(post_processing.emit_observables(raw_results)) == [[], []]


def test_measurements():
    @kernel(num_physical_qubits=2, aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        return gemini.logical.terminal_measure(q)

    post_processing = build_post_processing(main)

    # Simulate two shots
    raw_results = [[True, False, True, True], [True, False, False, True]]
    results = list(post_processing.emit_return(raw_results))
    assert results == [
        ilist.IList([ilist.IList([True, False]), ilist.IList([True, True])]),
        ilist.IList([ilist.IList([True, False]), ilist.IList([False, True])]),
    ]


def test_detectors():
    @kernel(num_physical_qubits=1, aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        m = gemini.logical.terminal_measure(q)
        return squin.set_detector(ilist.IList([m[0][0], m[1][0]]), [0, 1])

    post_processing = build_post_processing(main)
    # Simulate two shots
    raw_results = [[True, False], [True, True]]

    results = list(post_processing.emit_return(raw_results))
    assert results == [True, False]
    assert list(post_processing.emit_detectors(raw_results)) == [[True], [False]]
    assert list(post_processing.emit_observables(raw_results)) == [[], []]


def test_tuple():
    @kernel(num_physical_qubits=1, aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        m = gemini.logical.terminal_measure(q)
        return m, squin.set_detector(ilist.IList([m[0][0], m[1][0]]), [0, 1])

    post_processing = build_post_processing(main)
    # Simulate two shots
    raw_results = [[True, False], [True, True]]
    results = list(post_processing.emit_return(raw_results))
    assert results == [
        (ilist.IList([ilist.IList([True]), ilist.IList([False])]), True),
        (ilist.IList([ilist.IList([True]), ilist.IList([True])]), False),
    ]


def test_collects_detectors_and_observables_not_returned():
    @kernel(num_physical_qubits=1, aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        measurements = gemini.logical.terminal_measure(q)
        squin.set_detector(ilist.IList([measurements[0][0]]), [0, 0])
        squin.set_detector(
            ilist.IList([measurements[0][0], measurements[1][0]]), [0, 1]
        )
        squin.set_observable(ilist.IList([measurements[1][0]]))
        squin.set_observable(ilist.IList([measurements[0][0], measurements[1][0]]))

    post_processing = build_post_processing(main)
    raw_results = [[True, False], [True, True]]

    assert list(post_processing.emit_return(raw_results)) == [None, None]
    assert list(post_processing.emit_detectors(raw_results)) == [
        [True, True],
        [True, False],
    ]
    assert list(post_processing.emit_observables(raw_results)) == [
        [False, True],
        [True, False],
    ]
    assert all(
        type(value) is bool
        for shot in post_processing.emit_detectors(raw_results)
        for value in shot
    )


def test_empty_detector_reduces_to_false():
    @kernel(num_physical_qubits=1, aggressive_unroll=True)
    def main():
        q = squin.qalloc(1)
        gemini.logical.terminal_measure(q)
        return squin.set_detector(ilist.IList([]), [0, 0])

    post_processing = build_post_processing(main)

    assert list(post_processing.emit_return([[True]])) == [False]
    assert list(post_processing.emit_detectors([[True]])) == [[False]]


def test_logical_compile_uses_source_kernel_post_processing(monkeypatch):
    from bloqade.gemini.compile import compile_task

    @kernel(num_physical_qubits=1, aggressive_unroll=True)
    def main():
        q = squin.qalloc(1)
        return gemini.logical.terminal_measure(q)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("post-processing must not use the lowered move kernel")

    monkeypatch.setattr(
        atom.AtomInterpreter,
        "get_post_processing",
        fail_if_called,
        raising=False,
    )

    *_, post_processing = compile_task(main)

    assert post_processing is not None


def test_generate_post_processing_keeps_its_callable_contract():
    """The pre-existing entry point still hands back a bare callable over a
    2D array of raw measurements, not the three-emitter object."""

    @kernel(num_physical_qubits=2, aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        return gemini.logical.terminal_measure(q)

    func = generate_post_processing(main)
    assert func is not None
    assert not isinstance(func, atom.PostProcessing)

    # Each entry point takes the form its own signature declares -- ndarray for
    # the legacy callable, a nested sequence for the emitter -- and they agree.
    raw_results = [[True, False, True, True]]
    assert list(func(np.array(raw_results))) == list(
        build_post_processing(main).emit_return(raw_results)
    )


def test_generate_post_processing_returns_none_when_inference_fails(monkeypatch):
    """``build_post_processing`` raises where the older entry point returned
    None. The latter keeps its contract by mapping that back."""

    @kernel
    def main():
        return

    def _cannot_infer(mt):
        raise InterpreterError("Unable to infer return result value")

    monkeypatch.setattr(post_processing_module, "build_post_processing", _cannot_infer)
    assert generate_post_processing(main) is None
