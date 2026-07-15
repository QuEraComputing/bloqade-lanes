from __future__ import annotations

import gc
import sys
import weakref
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from kirin import ir
from kirin.dialects import func

from bloqade import squin
from bloqade.gemini.device import (
    AbstractSimulatorBackend,
    BackendSample,
    CliffTSimulatorBackend,
    PyQrackSimulatorBackend,
    TsimSimulatorBackend,
)
from bloqade.gemini.device.simulator_backend import (
    _clifft_compatible_stim_text,
    _get_tsim_circuit,
)


@squin.kernel
def _physical_kernel():
    reg = squin.qalloc(1)
    return squin.broadcast.measure(reg)


@squin.kernel
def _random_physical_kernel():
    reg = squin.qalloc(1)
    squin.h(reg[0])
    return squin.broadcast.measure(reg)


@squin.kernel
def _one_physical_kernel():
    reg = squin.qalloc(1)
    squin.x(reg[0])
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


def _fake_clifft(monkeypatch, *, sample_result=None):
    clifft = ModuleType("clifft")
    clifft.compile = MagicMock(return_value="program")  # type: ignore[attr-defined]
    clifft.sample = MagicMock(  # type: ignore[attr-defined]
        return_value=sample_result
        or SimpleNamespace(
            measurements=np.array([[0, 1]], dtype=np.uint8),
            detectors=np.array([[1]], dtype=np.uint8),
            observables=np.array([[0]], dtype=np.uint8),
        )
    )
    monkeypatch.setitem(sys.modules, "clifft", clifft)
    return clifft


def test_clifft_compatible_stim_text_strips_instruction_tags_only():
    assert (
        _clifft_compatible_stim_text(
            "I_ERROR[loss](0)\nDETECTOR[coords](1)\n# [comment]"
        )
        == "I_ERROR(0)\nDETECTOR(1)\n# [comment]"
    )


def test_clifft_backend_compiles_once_normalizes_measurements_and_passes_seed(
    monkeypatch,
):
    clifft = _fake_clifft(monkeypatch)
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "I_ERROR[loss](0)\nM 0"
    backend = CliffTSimulatorBackend(seed=123, tsim_backend=tsim_backend)

    first = backend.sample(_physical_kernel, shots=2)
    second = backend.sample(_physical_kernel, shots=2)

    assert first.measurements is not None
    assert first.measurements.dtype == np.bool_
    assert np.array_equal(first.measurements, [[False, True]])
    assert second.measurements is not None
    clifft.compile.assert_called_once_with("I_ERROR(0)\nM 0")  # type: ignore[attr-defined]
    assert clifft.sample.call_count == 2  # type: ignore[attr-defined]
    clifft.sample.assert_called_with(  # type: ignore[attr-defined]
        "program", shots=2, seed=123
    )


def test_clifft_backend_omits_missing_seed_and_returns_native_detectors(monkeypatch):
    clifft = _fake_clifft(monkeypatch)
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "M 0"
    backend = CliffTSimulatorBackend(tsim_backend=tsim_backend)

    sample = backend.sample(_physical_kernel, shots=2, run_detectors=True)

    clifft.sample.assert_called_once_with("program", shots=2)  # type: ignore[attr-defined]
    assert sample.detectors is not None
    assert sample.observables is not None
    assert np.array_equal(sample.detectors, [[True]])
    assert np.array_equal(sample.observables, [[False]])


def test_clifft_backend_delegates_tsim_circuit_and_dem_to_injected_backend():
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "circuit"
    tsim_backend.detector_error_model.return_value = "dem"
    backend = CliffTSimulatorBackend(tsim_backend=tsim_backend)

    assert _get_tsim_circuit(backend, _physical_kernel) == "circuit"
    assert backend.detector_error_model(_physical_kernel) == "dem"
    tsim_backend._tsim_circuit.assert_called_once_with(_physical_kernel)
    tsim_backend.detector_error_model.assert_called_once_with(_physical_kernel)


def test_clifft_backend_reframes_missing_tsim_dependency(monkeypatch):
    _fake_clifft(monkeypatch)
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.side_effect = ImportError("missing tsim")
    tsim_backend.detector_error_model.side_effect = ImportError("missing tsim")
    backend = CliffTSimulatorBackend(tsim_backend=tsim_backend)

    with pytest.raises(ImportError, match=r"CliffT.*bloqade-lanes\[sim\]"):
        backend.sample(_physical_kernel, shots=1)
    with pytest.raises(ImportError, match=r"CliffT.*bloqade-lanes\[sim\]"):
        backend.detector_error_model(_physical_kernel)


def test_clifft_program_cache_drops_dead_kernel(monkeypatch):
    _fake_clifft(monkeypatch)

    class Kernel:
        pass

    kernel = Kernel()
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "M 0"
    backend = CliffTSimulatorBackend(tsim_backend=tsim_backend)

    backend._clifft_program(cast(ir.Method, kernel))
    assert len(backend._programs) == 1
    tsim_backend.reset_mock()
    kernel_ref = weakref.ref(kernel)
    del kernel
    gc.collect()

    assert kernel_ref() is None
    assert len(backend._programs) == 0


def test_clifft_backend_real_seeded_sampling_and_dem():
    pytest.importorskip("clifft")
    pytest.importorskip("tsim")
    first_backend = CliffTSimulatorBackend(seed=991)
    second_backend = CliffTSimulatorBackend(seed=991)

    first = first_backend.sample(_random_physical_kernel, shots=16)
    second = second_backend.sample(_random_physical_kernel, shots=16)

    assert first.measurements is not None
    assert second.measurements is not None
    assert first.measurements.shape == (16, 1)
    assert np.array_equal(first.measurements, second.measurements)
    assert first_backend.detector_error_model(_random_physical_kernel) is not None
    first_backend._programs.clear()
    second_backend._programs.clear()


def test_pyqrack_backend_real_deterministic_sampling_uses_fresh_shot_tasks():
    pytest.importorskip("pyqrack")
    backend = PyQrackSimulatorBackend(seed=441)

    sample = backend.sample(_one_physical_kernel, shots=16)

    assert sample.measurements is not None
    assert sample.measurements.shape == (16, 1)
    assert sample.measurements.dtype == np.bool_
    assert np.all(sample.measurements)


def test_pyqrack_backend_prepares_owned_annotation_free_kernel_once(monkeypatch):
    from bloqade.decoders.dialects.annotate.stmts import SetDetector

    from bloqade.gemini.device.physical_simulator import (
        append_measurements_and_annotations_physical,
    )

    annotated = _physical_kernel.similar()
    append_measurements_and_annotations_physical(annotated, m2dets=[[1]], m2obs=None)
    source_ir = annotated.print_str()
    similar = MagicMock(wraps=annotated.similar)
    monkeypatch.setattr(annotated, "similar", similar)

    sample = PyQrackSimulatorBackend(seed=2).sample(
        annotated, shots=3, run_detectors=True
    )

    assert sample.measurements is not None
    assert sample.measurements.shape == (3, 1)
    assert similar.call_count == 1
    assert annotated.print_str() == source_ir
    assert any(
        isinstance(stmt, SetDetector) for stmt in annotated.callable_region.walk()
    )


def test_pyqrack_backend_delegates_tsim_circuit_and_dem_to_injected_backend():
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.return_value = "circuit"
    tsim_backend.detector_error_model.return_value = "dem"
    backend = PyQrackSimulatorBackend(tsim_backend=tsim_backend)

    assert _get_tsim_circuit(backend, _physical_kernel) == "circuit"
    assert backend.detector_error_model(_physical_kernel) == "dem"
    tsim_backend._tsim_circuit.assert_called_once_with(_physical_kernel)
    tsim_backend.detector_error_model.assert_called_once_with(_physical_kernel)


def test_pyqrack_backend_reframes_missing_tsim_dependency():
    tsim_backend = MagicMock(spec=TsimSimulatorBackend)
    tsim_backend._tsim_circuit.side_effect = ImportError("missing tsim")
    tsim_backend.detector_error_model.side_effect = ImportError("missing tsim")
    backend = PyQrackSimulatorBackend(tsim_backend=tsim_backend)

    with pytest.raises(ImportError, match=r"PyQrack.*bloqade-lanes\[sim\]"):
        _get_tsim_circuit(backend, _physical_kernel)
    with pytest.raises(ImportError, match=r"PyQrack.*bloqade-lanes\[sim\]"):
        backend.detector_error_model(_physical_kernel)


def test_pyqrack_measurements_are_mapped_explicitly():
    values = SimpleNamespace(Zero=0, One=1, Lost=2)

    assert PyQrackSimulatorBackend._measurement_to_bool(0, values) is False
    assert PyQrackSimulatorBackend._measurement_to_bool(1, values) is True
    with pytest.raises(ValueError, match="Unsupported PyQrack measurement"):
        PyQrackSimulatorBackend._measurement_to_bool(2, values)
