from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
from kirin import ir
from kirin.dialects import func

from bloqade import squin
from bloqade.gemini.device import (
    AbstractSimulatorBackend,
    BackendSample,
    TsimSimulatorBackend,
)
from bloqade.gemini.device.simulator_backend import _get_tsim_circuit


@squin.kernel
def _physical_kernel():
    reg = squin.qalloc(1)
    return squin.broadcast.measure(reg)


def test_backend_contract_has_exactly_two_abstract_operations():
    assert AbstractSimulatorBackend.__abstractmethods__ == {
        "sample",
        "detector_error_model",
    }


def test_backend_sample_defaults_to_no_payloads():
    assert BackendSample() == BackendSample(
        measurements=None, detectors=None, observables=None
    )


def test_tsim_circuit_cache_reuses_live_kernel_and_drops_dead_kernel(monkeypatch):
    class Kernel:
        def similar(self):
            return _physical_kernel.similar()

    circuit_factory = MagicMock(side_effect=lambda _: object())
    monkeypatch.setattr(
        "bloqade.gemini.device.simulator_backend._tsim",
        lambda: SimpleNamespace(Circuit=circuit_factory),
    )
    backend = TsimSimulatorBackend()
    first_kernel = Kernel()
    second_kernel = Kernel()

    first_circuit = backend._tsim_circuit(cast(ir.Method, first_kernel))
    assert backend._tsim_circuit(cast(ir.Method, first_kernel)) is first_circuit
    assert backend._tsim_circuit(cast(ir.Method, second_kernel)) is not first_circuit
    assert circuit_factory.call_count == 2
    assert len(backend._circuits) == 2

    first_ref = weakref.ref(first_kernel)
    del first_kernel
    gc.collect()

    assert first_ref() is None
    assert len(backend._circuits) == 1


def test_tsim_conversion_removes_return_from_owned_kernel(monkeypatch):
    converted = []
    monkeypatch.setattr(
        "bloqade.gemini.device.simulator_backend._tsim",
        lambda: SimpleNamespace(
            Circuit=lambda kernel: converted.append(kernel) or object()
        ),
    )
    source_ir = _physical_kernel.print_str()

    TsimSimulatorBackend()._tsim_circuit(_physical_kernel)

    assert _physical_kernel.print_str() == source_ir
    converted_return = converted[0].callable_region.blocks[0].last_stmt
    assert isinstance(converted_return, func.Return)
    assert isinstance(converted_return.value.owner, func.ConstantNone)


def _backend_with_circuit(circuit: MagicMock) -> TsimSimulatorBackend:
    backend = TsimSimulatorBackend()
    backend._tsim_circuit = MagicMock(return_value=circuit)  # type: ignore[method-assign]
    return backend


def test_tsim_clifford_measurement_sampling_uses_stim():
    circuit = MagicMock(is_clifford=True)
    circuit.stim_circuit.compile_sampler.return_value.sample.return_value = [[0, 1]]
    backend = _backend_with_circuit(circuit)

    sample = backend.sample(_physical_kernel, shots=1)

    assert sample.measurements is not None
    assert np.array_equal(sample.measurements, [[False, True]])
    circuit.stim_circuit.compile_sampler.assert_called_once_with()
    circuit.compile_sampler.assert_not_called()


def test_tsim_nonclifford_measurement_sampling_uses_tsim():
    circuit = MagicMock(is_clifford=False)
    circuit.compile_sampler.return_value.sample.return_value = [[1]]
    backend = _backend_with_circuit(circuit)

    sample = backend.sample(_physical_kernel, shots=1)

    assert sample.measurements is not None
    assert np.array_equal(sample.measurements, [[True]])
    circuit.compile_sampler.assert_called_once_with()
    circuit.stim_circuit.compile_sampler.assert_not_called()


def test_tsim_clifford_detector_sampling_uses_measurement_converter():
    circuit = MagicMock(is_clifford=True)
    measurements = np.array([[True, False]])
    circuit.stim_circuit.compile_sampler.return_value.sample.return_value = measurements
    converter = circuit.compile_m2d_converter.return_value
    converter.convert.return_value = (np.array([[1]]), np.array([[0]]))
    backend = _backend_with_circuit(circuit)

    sample = backend.sample(_physical_kernel, shots=1, run_detectors=True)

    circuit.compile_m2d_converter.assert_called_once_with(skip_reference_sample=True)
    converter.convert.assert_called_once_with(
        measurements=measurements, separate_observables=True
    )
    assert sample.detectors is not None
    assert sample.observables is not None
    assert np.array_equal(sample.detectors, [[True]])
    assert np.array_equal(sample.observables, [[False]])


def test_tsim_nonclifford_detector_sampling_uses_detector_sampler():
    circuit = MagicMock(is_clifford=False)
    sampler = circuit.compile_detector_sampler.return_value
    sampler.sample.return_value = (np.array([[0]]), np.array([[1]]))
    backend = _backend_with_circuit(circuit)

    sample = backend.sample(_physical_kernel, shots=1, run_detectors=True)

    sampler.sample.assert_called_once_with(shots=1, separate_observables=True)
    assert sample.detectors is not None
    assert sample.observables is not None
    assert np.array_equal(sample.detectors, [[False]])
    assert np.array_equal(sample.observables, [[True]])


def test_tsim_detector_error_model_and_structural_capability():
    circuit = MagicMock()
    circuit.detector_error_model.return_value = "dem"
    backend = _backend_with_circuit(circuit)

    assert _get_tsim_circuit(backend, _physical_kernel) is circuit
    assert backend.detector_error_model(_physical_kernel) == "dem"
    circuit.detector_error_model.assert_called_once_with(
        approximate_disjoint_errors=True
    )
