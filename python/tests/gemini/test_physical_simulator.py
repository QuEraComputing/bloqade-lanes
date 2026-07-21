from unittest.mock import MagicMock

import numpy as np
import pytest
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import func, ilist, ssacfg

import bloqade.gemini.device.physical_simulator as physical_simulator_module
from bloqade import squin, types as bloqade_types
from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.device.physical_simulator import (
    PhysicalResult,
    PhysicalSimulator,
    PhysicalSimulatorTask,
    append_measurements_and_annotations_physical,
)
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.logical_mvp import _find_qubit_ssas
from bloqade.lanes.transform import PhysicalPipeline


def test_physical_result_uses_post_processing():
    raw_measurements = [[True, False, True]]
    post_processing = MagicMock()
    post_processing.emit_return.return_value = ["return-value"]
    post_processing.emit_detectors.return_value = [[True, False]]
    post_processing.emit_observables.return_value = [[False]]

    result = PhysicalResult(
        _raw_measurements=raw_measurements,
        _detector_error_model=MagicMock(),
        _post_processing=post_processing,
        _fidelity_min=0.5,
        _fidelity_max=0.9,
    )

    assert result.measurements == raw_measurements
    assert result.return_values == ["return-value"]
    assert result.detectors == [[True, False]]
    assert result.observables == [[False]]
    post_processing.emit_return.assert_called_once_with(raw_measurements)
    post_processing.emit_detectors.assert_called_once_with(raw_measurements)
    post_processing.emit_observables.assert_called_once_with(raw_measurements)


def test_physical_task_run_clifford_returns_physical_result():
    task = MagicMock()
    task.tsim_circuit.is_clifford = True
    task.tsim_circuit.stim_circuit.compile_sampler.return_value.sample.return_value = (
        np.array([[True, False]])
    )
    task.fidelity_bounds.return_value = (0.5, 0.9)
    task.detector_error_model = MagicMock()
    task._post_processing = MagicMock()

    result = PhysicalSimulatorTask.run(task, shots=1, with_noise=True)

    task.tsim_circuit.stim_circuit.compile_sampler.assert_called_once_with()
    task.measurement_sampler.sample.assert_not_called()
    assert isinstance(result, PhysicalResult)
    assert result.measurements == [[True, False]]


def test_physical_simulator_exports_from_gemini_namespace():
    from bloqade.gemini import PhysicalSimulator
    from bloqade.gemini.device import PhysicalSimulator as DevicePhysicalSimulator

    assert PhysicalSimulator is DevicePhysicalSimulator


def test_physical_simulator_task_passes_placement_strategy(monkeypatch):
    import bloqade.lanes.transform as pipeline_module
    from bloqade.lanes.analysis import atom

    captured = {}
    move_kernel = MagicMock()
    move_kernel.dialects = "dialects"

    class FakePhysicalPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def emit(self, kernel, no_raise=True):
            captured["kernel"] = kernel
            captured["no_raise"] = no_raise
            return move_kernel

    class FakeAtomInterpreter:
        def __init__(self, dialects, arch_spec):
            captured["atom_dialects"] = dialects
            captured["atom_arch_spec"] = arch_spec

        def get_post_processing(self, kernel):
            captured["post_processing_kernel"] = kernel
            return "post_processing"

    def fake_append_measurements_and_annotations_physical(kernel, m2dets, m2obs):
        captured["annotated_kernel"] = kernel
        captured["m2dets"] = m2dets
        captured["m2obs"] = m2obs

    monkeypatch.setattr(pipeline_module, "PhysicalPipeline", FakePhysicalPipeline)
    monkeypatch.setattr(atom, "AtomInterpreter", FakeAtomInterpreter)
    monkeypatch.setattr(
        physical_simulator_module,
        "append_measurements_and_annotations_physical",
        fake_append_measurements_and_annotations_physical,
    )

    kernel = MagicMock()
    placement_strategy = MagicMock()
    m2dets = [[1]]
    m2obs = [[1]]

    task = PhysicalSimulator().task(
        kernel,
        placement_strategy=placement_strategy,
        m2dets=m2dets,
        m2obs=m2obs,
    )

    assert captured["placement_strategy"] is placement_strategy
    assert captured["annotated_kernel"] is kernel
    assert captured["m2dets"] is m2dets
    assert captured["m2obs"] is m2obs
    assert captured["kernel"] is kernel
    assert captured["no_raise"] is False
    assert captured["atom_dialects"] == "dialects"
    assert captured["post_processing_kernel"] is move_kernel
    assert task.physical_move_kernel is move_kernel
    assert task._post_processing == "post_processing"


def test_append_measurements_and_annotations_physical_preserves_kernel_return():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(4)
        measurements = squin.broadcast.measure(reg)
        return measurements

    append_measurements_and_annotations_physical(
        kernel,
        m2dets=[[1], [1]],
        m2obs=[[1], [0]],
    )

    assert sum(isinstance(s, SetDetector) for s in kernel.callable_region.walk()) == 2
    assert sum(isinstance(s, SetObservable) for s in kernel.callable_region.walk()) == 2

    arch_spec = get_arch_spec()
    physical_move_kernel = PhysicalPipeline(arch_spec=arch_spec).emit(
        kernel, no_raise=False
    )
    post_processing = atom.AtomInterpreter(
        physical_move_kernel.dialects, arch_spec=arch_spec
    ).get_post_processing(physical_move_kernel)

    raw_shots = [[True, False, False, True]]

    return_values = list(post_processing.emit_return(raw_shots))
    assert len(return_values) == 1
    assert list(return_values[0]) == raw_shots[0]
    assert list(post_processing.emit_detectors(raw_shots)) == [[True, True]]
    assert list(post_processing.emit_observables(raw_shots)) == [[True, False]]


def test_append_measurements_and_annotations_physical_uses_logical_mvp_helpers(
    monkeypatch,
):
    import bloqade.lanes.logical_mvp as logical_mvp

    find_qubit_ssas = MagicMock(wraps=logical_mvp._find_qubit_ssas)
    find_return_stmt = MagicMock(wraps=logical_mvp._find_return_stmt)
    insert_before = MagicMock(wraps=logical_mvp._insert_before)
    monkeypatch.setattr(logical_mvp, "_find_qubit_ssas", find_qubit_ssas)
    monkeypatch.setattr(logical_mvp, "_find_return_stmt", find_return_stmt)
    monkeypatch.setattr(logical_mvp, "_insert_before", insert_before)

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.broadcast.measure(reg)

    append_measurements_and_annotations_physical(
        kernel,
        m2dets=[[1], [1]],
        m2obs=None,
    )

    find_qubit_ssas.assert_called_once_with(kernel)
    find_return_stmt.assert_called_once_with(kernel)
    assert insert_before.call_count > 0


def test_append_measurements_and_annotations_physical_validates_block_size():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.broadcast.measure(reg)

    with pytest.raises(
        ValueError,
        match="physical qubits must be divisible",
    ):
        append_measurements_and_annotations_physical(
            kernel,
            m2dets=[[1], [1]],
            m2obs=None,
        )


def test_append_measurements_and_annotations_physical_accepts_new_at_allocations():
    kernel = squin.kernel.add(gemini_qubit)
    kernel.run_pass = squin.kernel.run_pass

    @kernel(typeinfer=True)
    def pinned_kernel():
        q0 = gemini_qubit.new_at(0, 0, 0)
        q1 = gemini_qubit.new_at(0, 1, 0)
        squin.broadcast.measure(ilist.IList([q0, q1]))

    append_measurements_and_annotations_physical(
        pinned_kernel,
        m2dets=[[1], [1]],
        m2obs=[[1], [0]],
    )

    assert (
        sum(isinstance(s, SetDetector) for s in pinned_kernel.callable_region.walk())
        == 1
    )
    assert (
        sum(isinstance(s, SetObservable) for s in pinned_kernel.callable_region.walk())
        == 1
    )


def test_find_qubit_ssas_ignores_qubit_typed_results_without_addresses():
    test_dialect = ir.Dialect("test.qubit_alloc")

    @statement(dialect=test_dialect)
    class CustomQubitAlloc(ir.Statement):
        result: ir.ResultValue = info.result(bloqade_types.QubitType)

    block = ir.Block(argtypes=(types.MethodType,))
    alloc = CustomQubitAlloc()
    none_stmt = func.ConstantNone()
    for stmt in (alloc, none_stmt, func.Return(none_stmt.result)):
        block.stmts.append(stmt)

    function = func.Function(
        sym_name="custom_alloc",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=ir.Region(blocks=block),
    )
    method = ir.Method(
        dialects=ir.DialectGroup([func.dialect, ssacfg.dialect, test_dialect]),
        code=function,
        sym_name="custom_alloc",
        arg_names=[],
    )

    assert _find_qubit_ssas(method) == []
