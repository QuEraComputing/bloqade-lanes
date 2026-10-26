"""TaskBuilder-to-local-simulator integration with the existing result API."""

import json
from concurrent.futures import Future
from inspect import Parameter, signature
from threading import Event, enumerate as enumerate_threads
from time import monotonic, sleep
from uuid import UUID, uuid4

import numpy as np
import pytest
from bloqade.core.device import DictStorage, ShotFilter
from bloqade.core.device.future import ApiFetchOptions
from bloqade.core.device.local_storage import SQLiteStorage
from bloqade.core.device.task_builder import TaskBuilder, TaskFinalizeError
from kirin import ir
from kirin.dialects import ilist
from kirin.serialization import JSONSerializer
from kirin.validation import ValidationSuite

from bloqade import squin
from bloqade.gemini import (
    GeminiLogicalDevice,
    GeminiLogicalResult,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorDevice,
    logical,
)
from bloqade.gemini.device import logical as device_logical
from bloqade.gemini.device.logical import simulator_device as device_module
from bloqade.gemini.device.simulator_backend import (
    BackendSample,
    PPVMSimulatorBackend,
    TsimSimulatorBackend,
)


@logical.kernel
def pinned():
    q = logical.qalloc_at(ilist.IList([8, 1]))
    squin.x(q[0])
    return logical.terminal_measure(q)


@logical.kernel
def bell():
    q = logical.qalloc_at(ilist.IList([4, 5]))
    squin.h(q[0])
    squin.cx(q[0], q[1])
    return logical.terminal_measure(q)


@logical.kernel
def all_slots():
    q = logical.qalloc_at(ilist.IList([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
    return logical.terminal_measure(q)


@logical.kernel
def annotated():
    q = logical.qalloc_at(ilist.IList([9]))
    measurements = logical.terminal_measure(q)
    squin.set_detector(measurements[0], [0])
    squin.set_observable(measurements[0])
    return measurements[0][0], measurements[0]


@logical.kernel(verify=False)
def parameterized(theta: float):
    q = squin.qalloc(1)
    squin.rx(theta, q[0])
    return logical.terminal_measure(q)


def builder_for(kernel: ir.Method = pinned, shots: int = 3) -> TaskBuilder:
    builder = TaskBuilder()
    builder.add_subtask(kernel, shots, metadata={"name": kernel.sym_name})
    return builder


def test_public_exports_are_the_same_class():
    assert device_logical.GeminiLogicalSimulatorDevice is GeminiLogicalSimulatorDevice


def test_finalize_context_defaults_match_hardware_device():
    device = GeminiLogicalSimulatorDevice()
    context = device._finalize_context()
    hardware = GeminiLogicalDevice()._finalize_context()
    assert context.program_language == hardware.program_language
    assert context.language_version == hardware.language_version
    assert type(context.kernel_serializer) is type(hardware.kernel_serializer)
    assert context.dialect_group is hardware.dialect_group
    assert isinstance(context.validation_suite, ValidationSuite)
    assert isinstance(hardware.validation_suite, ValidationSuite)
    assert context.validation_suite.passes == hardware.validation_suite.passes
    assert context.validation_suite is not hardware.validation_suite


def test_custom_finalize_context_is_used_by_submission(monkeypatch):
    serializer = JSONSerializer()
    suite = ValidationSuite([])
    device = GeminiLogicalSimulatorDevice(
        program_language="custom-squin",
        language_version="1.2.3",
        kernel_serializer=serializer,
        dialect_group=logical.kernel,
        validation_suite=suite,
    )
    context = device._finalize_context()
    assert context.program_language == "custom-squin"
    assert context.language_version == "1.2.3"
    assert context.kernel_serializer is serializer
    assert context.dialect_group is logical.kernel
    assert context.validation_suite is suite
    contexts = []
    finalize = TaskBuilder._finalize

    def record(builder, ctx):
        contexts.append(ctx)
        return finalize(builder, ctx)

    def forbidden(*args, **kwargs):
        pytest.fail("simulator must use its own finalization context")

    monkeypatch.setattr(TaskBuilder, "_finalize", record)
    monkeypatch.setattr(GeminiLogicalDevice, "_finalize_context", forbidden)
    result = device.run_async(builder_for(shots=1), dry_run=False).result(60)
    assert contexts == [context]
    task_id = next(iter(result.task_ids()))
    assert result.storage.get_program_language(task_id) == "custom-squin.v1.2.3"
    program = next(iter(result.storage.get_programs(task_ids=(task_id,))))
    assert json.loads(program["content"])["version"] == "1.2.3"


@pytest.mark.parametrize("as_string", [False, True])
def test_submission_uses_supplied_storage_and_profile_metadata(as_string):
    storage = DictStorage()
    profile = uuid4()
    profile_arg = str(profile) if as_string else profile
    result = (
        GeminiLogicalSimulatorDevice(seed=12)
        .run_async(
            builder_for(shots=1), dry_run=False, storage=storage, profile_id=profile_arg
        )
        .result(60)
    )
    assert result.storage is storage
    task_id = next(iter(result.task_ids()))
    assert storage.get_profile_id(task_id) == profile
    assert result.shot_results()[0].shape == (1, 160)


def test_definition_profile_takes_precedence_over_submission_profile(monkeypatch):
    profile = uuid4()
    finalize = TaskBuilder._finalize

    def with_profile(builder, ctx):
        return finalize(builder, ctx).model_copy(update={"profile_id": profile})

    monkeypatch.setattr(TaskBuilder, "_finalize", with_profile)
    result = (
        GeminiLogicalSimulatorDevice()
        .run_async(builder_for(shots=1), dry_run=False, profile_id=uuid4())
        .result(60)
    )
    task_id = next(iter(result.task_ids()))
    assert result.storage.get_profile_id(task_id) == profile


def test_remote_submission_options_are_accepted_and_ignored(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local simulation must not resolve groups or authenticate")

    monkeypatch.setattr(GeminiLogicalDevice, "_resolve_group_id", forbidden)
    monkeypatch.setattr(GeminiLogicalDevice, "authenticate", forbidden)
    result = (
        GeminiLogicalSimulatorDevice(seed=12)
        .run_async(
            builder_for(shots=1),
            dry_run=False,
            group="no-local-group-resolution",
            fetch_options=ApiFetchOptions(shots_per_fetch=1, poll_interval_initial=999),
        )
        .result(60)
    )
    task_id = next(iter(result.task_ids()))
    assert result.storage.get_task_group_id(task_id) is None
    assert result.shot_results()[0].shape == (1, 160)


def test_device_signature_requires_explicit_dry_run():
    assert (
        signature(GeminiLogicalSimulatorDevice.run_async).parameters["dry_run"].default
        is Parameter.empty
    )


def test_dry_run_with_submission_options_does_not_write_storage(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not start a worker or write storage")

    device = GeminiLogicalSimulatorDevice(seed=12)
    storage = DictStorage()
    state = json.dumps(device._seed_rng.bit_generator.state)
    monkeypatch.setattr(device_module, "ThreadPoolExecutor", forbidden)
    monkeypatch.setattr(storage, "add_task_definition", forbidden)
    monkeypatch.setattr(storage, "add_shots", forbidden)
    assert (
        device.run_async(
            builder_for(),
            dry_run=True,
            group="ignored",
            profile_id=UUID(int=1),
            storage=storage,
            fetch_options=ApiFetchOptions(shots_per_fetch=1),
        )
        is None
    )
    assert not storage.task_ids()
    assert json.dumps(device._seed_rng.bit_generator.state) == state


def test_sqlite_storage_rejected_before_starting_worker(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported storage must be rejected before starting a worker")

    device = GeminiLogicalSimulatorDevice(seed=12)
    state = json.dumps(device._seed_rng.bit_generator.state)
    monkeypatch.setattr(device_module, "ThreadPoolExecutor", forbidden)
    with SQLiteStorage(":memory:") as storage:
        with pytest.raises(TypeError, match="SQLiteStorage.*not supported"):
            device.run_async(builder_for(shots=1), dry_run=False, storage=storage)
        assert not storage.task_ids()
    assert json.dumps(device._seed_rng.bit_generator.state) == state


def test_dry_run_accepts_sqlite_storage_without_accessing_it(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not access storage or start a worker")

    monkeypatch.setattr(device_module, "ThreadPoolExecutor", forbidden)
    with SQLiteStorage(":memory:") as storage:
        monkeypatch.setattr(storage, "add_task_definition", forbidden)
        monkeypatch.setattr(storage, "add_shots", forbidden)
        assert (
            GeminiLogicalSimulatorDevice().run_async(
                builder_for(), dry_run=True, storage=storage
            )
            is None
        )
        assert not storage.task_ids()


@pytest.mark.parametrize("backend", ["tsim", "ppvm"])
def test_real_backend_result_frames_and_postprocessing(backend):
    pytest.importorskip("bloqade.tsim")
    if backend == "ppvm":
        pytest.importorskip("ppvm")
    future = GeminiLogicalSimulatorDevice(backend=backend, seed=1234).run_async(
        builder_for(), dry_run=False
    )
    assert type(future) is Future
    result = future.result(timeout=60)
    assert type(result) is GeminiLogicalResult
    assert [shots.shape for shots in result.shot_results()] == [(3, 160)]
    values = np.asarray(result.return_values[0], dtype=bool)
    assert values.shape == (3, 2, 7)
    assert np.all(np.logical_xor.reduce(values, axis=2) == [True, False])
    assert np.asarray(result.measurements[0]).shape == (3, 14)
    assert np.all(result.filling_at_start[0])
    assert result.postselect_on_fully_filled().shot_results()[0].shape == (3, 160)
    sorted_shots = list(
        result.storage.get_shots(
            shot_filter=ShotFilter(
                task_ids=tuple(result.task_ids()), frame_type="SORTED"
            )
        )
    )
    expected = set(range(128, 135)) | set(range(16, 23))
    assert all(set(np.flatnonzero(shot.bitstring)) == expected for shot in sorted_shots)
    assert all(shot.bitstring.dtype == bool for shot in sorted_shots)
    assert all(
        subtask["completed_date"] is not None for subtask in result.full_subtasks()
    )


def test_detectors_observables_and_structured_returns_use_existing_result():
    result = (
        GeminiLogicalSimulatorDevice(seed=123)
        .run_async(builder_for(annotated, 3), dry_run=False)
        .result(60)
    )
    raw = np.asarray(result.measurements[0])
    parity = np.logical_xor.reduce(raw, axis=1)
    np.testing.assert_array_equal(np.asarray(result.detectors[0])[:, 0], parity)
    np.testing.assert_array_equal(np.asarray(result.observables[0])[:, 0], parity)
    for returned, measurements in zip(result.return_values[0], raw):
        assert returned[0] == measurements[0]
        np.testing.assert_array_equal(returned[1], measurements)


def test_multiple_programs_repeated_subtasks_and_metadata(monkeypatch):
    calls = []
    compile_task = GeminiLogicalSimulator.task

    def record(self, kernel):
        calls.append(kernel.sym_name)
        return compile_task(self, kernel)

    monkeypatch.setattr(GeminiLogicalSimulator, "task", record)
    builder = builder_for(shots=3)
    builder.add_subtask(bell, 5, metadata={"name": "bell"})
    builder.add_subtask(pinned, 2, metadata={"name": "repeat"})
    result = (
        GeminiLogicalSimulatorDevice(seed=12, batch_shots=2)
        .run_async(builder, dry_run=False)
        .result(60)
    )
    assert calls == ["pinned", "bell"]
    assert [shots.shape for shots in result.shot_results()] == [
        (3, 160),
        (5, 160),
        (2, 160),
    ]
    assert [row["program_index"] for row in result.full_subtasks()] == [0, 1, 0]
    assert [
        json.loads(row["metadata"]["user_metadata"])["name"]
        for row in result.full_subtasks()
    ] == ["pinned", "bell", "repeat"]
    rows = list(result.storage.get_shots(shot_filter=result.shot_filter))
    assert [row.shot_index for row in rows] == list(range(10))
    assert [row.subtask_shot_index for row in rows] == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]
    bell_values = np.logical_xor.reduce(np.asarray(result.return_values[1]), axis=2)
    assert np.all(bell_values[:, 0] == bell_values[:, 1])


def test_all_logical_slots_have_the_correct_physical_filling():
    result = (
        GeminiLogicalSimulatorDevice(seed=123)
        .run_async(builder_for(all_slots, 1), dry_run=False)
        .result(60)
    )
    sorted_shot = next(
        iter(result.storage.get_shots(shot_filter=ShotFilter(frame_type="SORTED")))
    )
    expected = {16 * slot + atom for slot in range(10) for atom in range(7)}
    assert set(np.flatnonzero(sorted_shot.bitstring)) == expected
    assert np.asarray(result.measurements[0]).shape == (1, 70)
    assert np.all(result.filling_at_start[0])


def test_dry_run_has_no_worker_and_does_not_consume_seed(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run must not start a worker or sample")

    device = GeminiLogicalSimulatorDevice(seed=12)
    state = json.dumps(device._seed_rng.bit_generator.state)
    monkeypatch.setattr(device_module, "ThreadPoolExecutor", forbidden)
    assert device.run_async(builder_for(), dry_run=True) is None
    assert "pinned" in capsys.readouterr().out
    assert json.dumps(device._seed_rng.bit_generator.state) == state


def test_simulation_does_not_authenticate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("local simulation must not authenticate")

    monkeypatch.setattr(GeminiLogicalDevice, "authenticate", forbidden)
    result = (
        GeminiLogicalSimulatorDevice(seed=12)
        .run_async(builder_for(shots=1), dry_run=False)
        .result(60)
    )
    assert result.shot_results()[0].shape == (1, 160)


def test_submission_snapshots_mutable_builder_and_releases_executor(monkeypatch):
    entered, proceed = Event(), Event()
    run_task = device_module._run_task
    threads_before = {thread.ident for thread in enumerate_threads()}

    def blocked(*args):
        entered.set()
        assert proceed.wait(60)
        return run_task(*args)

    monkeypatch.setattr(device_module, "_run_task", blocked)
    builder = builder_for(shots=1)
    device = GeminiLogicalSimulatorDevice(seed=1)
    future = device.run_async(builder, dry_run=False)
    try:
        assert entered.wait(60)
        builder.add_subtask(bell, 2)
        metadata = builder._subtasks[0].qlam_subtask.subtask_metadata
        assert metadata is not None
        metadata.user_metadata = '{"changed":true}'
    finally:
        proceed.set()
    result = future.result(60)
    assert len(result.full_subtasks()) == 1
    assert json.loads(result.full_subtasks()[0]["metadata"]["user_metadata"]) == {
        "name": "pinned"
    }
    workers = []
    deadline = monotonic() + 5
    while monotonic() < deadline:
        workers = [
            thread
            for thread in enumerate_threads()
            if thread.ident not in threads_before
            and thread.name.startswith("gemini-logical-simulator")
        ]
        if not workers:
            break
        sleep(0.01)
    assert not workers


def test_seeded_devices_replay_the_same_submission_sequence():
    a = GeminiLogicalSimulatorDevice(seed=123, batch_shots=2)
    b = GeminiLogicalSimulatorDevice(seed=123, batch_shots=2)
    builder = builder_for(bell, 5)
    for _ in range(2):
        ra = a.run_async(builder, dry_run=False).result(60)
        rb = b.run_async(builder, dry_run=False).result(60)
        np.testing.assert_array_equal(ra.shot_results()[0], rb.shot_results()[0])
        assert ra.task_ids() != rb.task_ids()


def test_backend_failure_is_raised_by_the_standard_future(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("sample failed")

    monkeypatch.setattr(TsimSimulatorBackend, "sample", fail)
    future = GeminiLogicalSimulatorDevice().run_async(
        builder_for(shots=1), dry_run=False
    )
    with pytest.raises(RuntimeError, match="sample failed"):
        future.result(60)
    assert future.done()


@pytest.mark.parametrize("payload", [None, np.zeros((1, 1), dtype=bool)])
def test_invalid_sample_shapes_are_not_stored(monkeypatch, payload):
    monkeypatch.setattr(
        TsimSimulatorBackend,
        "sample",
        lambda *args, **kwargs: BackendSample(measurements=payload),
    )
    future = GeminiLogicalSimulatorDevice().run_async(
        builder_for(shots=1), dry_run=False
    )
    with pytest.raises(ValueError, match="measurements|Measurement shape"):
        future.result(60)


def test_ppvm_loss_becomes_absence_in_detected_frame(monkeypatch):
    def all_lost(*args, shots):
        return BackendSample(measurements=np.full((shots, 14), None, dtype=object))

    monkeypatch.setattr(PPVMSimulatorBackend, "sample", all_lost)
    result = (
        GeminiLogicalSimulatorDevice(backend="ppvm")
        .run_async(builder_for(), dry_run=False)
        .result(60)
    )
    assert not np.any(result.shot_results()[0])
    assert np.all(result.measurements[0])
    assert np.all(result.filling_at_start[0])


def test_noisy_ppvm_sampling_uses_existing_noise_model():
    pytest.importorskip("ppvm")
    pytest.importorskip("bloqade.tsim")
    result = (
        GeminiLogicalSimulatorDevice(backend="ppvm", seed=123, with_noise=True)
        .run_async(builder_for(shots=3), dry_run=False)
        .result(60)
    )
    assert result.shot_results()[0].shape == (3, 160)
    assert np.all(result.filling_at_start[0])


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**63},
        {"batch_shots": 0},
        {"batch_shots": True},
        {"backend": "unknown"},
        {"with_noise": "true"},
    ],
)
def test_invalid_device_configuration(change):
    with pytest.raises((ValueError, TypeError)):
        GeminiLogicalSimulatorDevice(**change)


def test_empty_builder_rejected():
    with pytest.raises(ValueError, match="programs|subtasks"):
        GeminiLogicalSimulatorDevice().run_async(TaskBuilder(), dry_run=False)


def test_runtime_arguments_rejected_before_submission():
    builder = TaskBuilder()
    builder.add_subtask(parameterized, 1, theta=0.1)
    with pytest.raises(
        (TaskFinalizeError, ValueError), match="argument|parameter|validation"
    ):
        GeminiLogicalSimulatorDevice().run_async(builder, dry_run=False)
