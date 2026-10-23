"""Local TaskBuilder execution with hardware-compatible Gemini logical results."""

from __future__ import annotations

from _thread import LockType
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import uuid4

import numpy as np
from bloqade.core.device import DictStorage, ShotFilter, ShotResult
from bloqade.core.device.task_builder import TaskBuilder
from kirin.serialization import JSONSerializer

from .device import GeminiLogicalDevice
from .result import GeminiLogicalResult
from .utils import ShotRemappingException

if TYPE_CHECKING:
    from qlam_core.plugins.tasks.api.tasks_models import TaskDefinition

    from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

    from ..simulator import GeminiLogicalSimulatorTask
    from ..simulator_backend import AbstractSimulatorBackend


def _slm_layout(task: GeminiLogicalSimulatorTask) -> tuple[list[int], np.ndarray]:
    # These imports must stay lazy: the public Gemini package is also imported
    # while the Lanes analysis/transform modules are being initialized.
    from bloqade.lanes.analysis.atom import AtomInterpreter
    from bloqade.lanes.analysis.atom._shot_remapping import ShotRemappingErr
    from bloqade.lanes.bytecode.encoding import ZoneAddress
    from bloqade.lanes.dialects import move

    arch = task.physical_arch_spec
    zone = ZoneAddress(0)
    width = len(list(arch.yield_zone_locations(zone)))
    remapping = AtomInterpreter(
        task.physical_move_kernel.dialects, arch_spec=arch
    ).get_shot_remapping(task.physical_move_kernel)
    if isinstance(remapping, ShotRemappingErr):
        raise ShotRemappingException(
            f"Cannot map physical measurements: {remapping.diagnostic}"
        )
    mapping = remapping.mapping
    if (
        not mapping
        or remapping.frame_size != width
        or len(set(mapping)) != len(mapping)
        or any(index < 0 or index >= width for index in mapping)
    ):
        raise ValueError("Measurements must map one-to-one to the Zone-0 SLM frame")

    fills = [
        stmt
        for stmt in task.physical_move_kernel.callable_region.walk()
        if isinstance(stmt, move.Fill)
    ]
    if len(fills) != 1:
        raise ValueError("Only one initial SLM Fill is supported; no dynamic refills")
    sorted_frame = np.zeros(width, dtype=bool)
    for address in fills[0].location_addresses:
        if address.zone_id != 0:
            raise ValueError("Only SLM filling in Zone 0 is supported")
        index = arch.get_zone_index(address, zone)
        if index is None or not 0 <= index < width:
            raise ValueError("Initial filling has no corresponding Zone-0 SLM site")
        sorted_frame[index] = True
    return mapping, sorted_frame


def _run_task(
    definition: TaskDefinition,
    backend: AbstractSimulatorBackend,
    with_noise: bool,
    noise_model: LogicalNoiseModelABC | None,
    batch_shots: int,
) -> GeminiLogicalResult[Any]:
    from bloqade.gemini import logical

    from ..simulator import GeminiLogicalSimulator, _default_noise_model

    simulator = GeminiLogicalSimulator(
        backend=backend,
        noise_model=noise_model if noise_model is not None else _default_noise_model(),
    )
    serializer = JSONSerializer()
    prepared = []
    for program in definition.programs:
        kernel = logical.kernel.decode(serializer.decode(program.content))
        if kernel.args:
            raise ValueError("Simulator device programs must have no runtime arguments")
        # Kirin JSON omits analysis hints needed by explicitly pinned qalloc_at.
        run_pass = logical.kernel.run_pass
        if run_pass is None:
            raise RuntimeError("Gemini logical kernel has no pass pipeline")
        run_pass(kernel, aggressive_unroll=True, verify=False)
        task = simulator.task(kernel)
        mapping, sorted_frame = _slm_layout(task)
        physical_kernel = (
            task.physical_squin_kernel
            if with_noise
            else task.noiseless_physical_squin_kernel
        )
        prepared.append((physical_kernel, mapping, sorted_frame))

    storage = DictStorage()
    task_id = str(uuid4())
    storage.add_task_definition(task_id, definition, datetime.now(timezone.utc))
    shot_index = 0
    for subtask_index, subtask in enumerate(definition.subtasks):
        physical_kernel, mapping, sorted_frame = prepared[subtask.program_index]
        for start in range(0, subtask.num_shots, batch_shots):
            count = min(batch_shots, subtask.num_shots - start)
            sample = backend.sample(physical_kernel, shots=count)
            if sample.measurements is None:
                raise ValueError(
                    "The simulator backend must return physical measurements"
                )
            raw = np.asarray(sample.measurements)
            if raw.shape != (count, len(mapping)):
                raise ValueError(
                    f"Measurement shape {raw.shape} != {(count, len(mapping))}"
                )
            # PPVM represents atom loss as None. Like |1>, loss means absence
            # in DETECTED; normalize it before inverting measurement bits.
            measurements = np.asarray(np.where(raw == None, True, raw), dtype=bool)
            detected = np.zeros((count, len(sorted_frame)), dtype=bool)
            detected[:, mapping] = ~measurements
            rows = []
            for offset, detected_frame in enumerate(detected):
                for frame_type, bitstring in (
                    ("SORTED", sorted_frame),
                    ("DETECTED", detected_frame),
                ):
                    rows.append(
                        ShotResult(
                            task_id=task_id,
                            shot_index=shot_index + offset,
                            subtask_index=subtask_index,
                            subtask_shot_index=start + offset,
                            frame_type=frame_type,
                            bitstring=bitstring.copy(),
                        )
                    )
            storage.add_shots(rows)
            shot_index += count
        storage.update_subtasks_completed_date(
            task_id,
            [
                {
                    "completed_date": datetime.now(timezone.utc),
                    "shot_results": [{"subtask_index": subtask_index}],
                }
            ],
        )
    return GeminiLogicalResult(
        storage=storage,
        shot_filter=ShotFilter(task_ids=(task_id,), frame_type="DETECTED"),
    )


@dataclass(kw_only=True)
class GeminiLogicalSimulatorDevice:
    """Run TaskBuilder workloads locally and return GeminiLogicalResult views.

    Uses the same builder finalization and validation as GeminiLogicalDevice.
    Each submission returns a standard concurrent.futures.Future; its result is
    an existing GeminiLogicalResult backed by a fresh DictStorage. No QLAM
    server, authentication, context manager, or explicit close is required.
    Each submission's executor stops automatically after its work finishes.

    Sorting is ideal, and raw frames use physical Zone-0 SLM order (160 sites
    on the default architecture). Runtime arguments, dynamic refills and
    multiple SLM zones are unsupported. All result frames remain in memory;
    batching bounds temporary sampling arrays, not the size of the result.

    Args:
        backend: Sampling backend, "tsim" (default) or "ppvm". Install the
            optional ``bloqade-lanes[sim]`` dependencies to run simulations.
        seed: Optional root seed. Submissions receive successive child seeds;
            replay requires the same submission order and batching.
        with_noise: Insert circuit noise, default False. Sorting remains ideal.
        noise_model: Optional custom logical noise model. None uses the same
            default model as GeminiLogicalSimulator.
        batch_shots: Maximum shots per sampling batch, default 256.
    """

    backend: Literal["tsim", "ppvm"] = "tsim"
    seed: int | None = None
    with_noise: bool = False
    noise_model: LogicalNoiseModelABC | None = None
    batch_shots: int = 256
    _seed_rng: np.random.Generator = field(init=False, repr=False)
    _seed_lock: LockType = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        from ..simulator_backend import _validate_seed

        if self.backend not in ("tsim", "ppvm"):
            raise ValueError("backend must be 'tsim' or 'ppvm'")
        _validate_seed(self.seed)
        if not isinstance(self.with_noise, bool):
            raise TypeError("with_noise must be bool")
        if (
            isinstance(self.batch_shots, bool)
            or not isinstance(self.batch_shots, int)
            or self.batch_shots <= 0
        ):
            raise ValueError("batch_shots must be a positive integer")
        self._seed_rng = np.random.default_rng(self.seed)

    @overload
    def run_async(
        self, task_builder: TaskBuilder, *, dry_run: Literal[False] = False
    ) -> Future[GeminiLogicalResult[Any]]: ...

    @overload
    def run_async(
        self, task_builder: TaskBuilder, *, dry_run: Literal[True]
    ) -> None: ...

    @overload
    def run_async(
        self, task_builder: TaskBuilder, *, dry_run: bool
    ) -> Future[GeminiLogicalResult[Any]] | None: ...

    def run_async(
        self, task_builder: TaskBuilder, *, dry_run: bool = False
    ) -> Future[GeminiLogicalResult[Any]] | None:
        """Finalize a builder and execute a snapshot in a background thread.

        A dry run validates, serializes, and prints the builder without starting
        an executor or consuming a seed. Builder edits after submission do not
        affect the submitted task. Validation errors are raised immediately;
        compilation and sampling errors are raised by ``future.result()``.
        Standard future timeouts do not cancel a running simulation.
        """
        from ..simulator_backend import PPVMSimulatorBackend, TsimSimulatorBackend

        if not isinstance(task_builder, TaskBuilder):
            raise TypeError("run_async requires a TaskBuilder")
        definition = task_builder._finalize(GeminiLogicalDevice()._finalize_context())
        if not definition.programs or not definition.subtasks:
            raise ValueError("At least one program and subtask are required")
        for subtask in definition.subtasks:
            if not 0 <= subtask.program_index < len(definition.programs):
                raise ValueError("Invalid subtask program index")
            if subtask.arguments:
                raise ValueError("Runtime subtask arguments are unsupported")
            if subtask.num_shots <= 0:
                raise ValueError("num_shots must be positive")
        if dry_run:
            print(task_builder.summary())
            return None

        with self._seed_lock:
            seed = (
                int(self._seed_rng.integers(0, 2**63))
                if self.seed is not None
                else None
            )
        backend_cls = (
            TsimSimulatorBackend if self.backend == "tsim" else PPVMSimulatorBackend
        )
        backend = backend_cls(seed=seed)
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gemini-logical-simulator"
        )
        try:
            return executor.submit(
                _run_task,
                definition.model_copy(deep=True),
                backend,
                self.with_noise,
                self.noise_model,
                self.batch_shots,
            )
        finally:
            # shutdown(wait=False) lets the submitted task finish, then releases
            # the worker even if the device/notebook remains alive indefinitely.
            executor.shutdown(wait=False)
