from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from kirin import ir, rewrite

from bloqade import squin, tsim
from bloqade.gemini.device import GeminiLogicalSimulatorTask
from bloqade.lanes import GeminiLogicalSimulator

from ..standard.types import KirinKernel, MeasurementMap, SquinKernel, TsimCircuit
from .tasks import DemoTask, _ObservableFrame

_NONUNITARY_PREFIXES = (
    "M",
    "MX",
    "MY",
    "MR",
    "MRX",
    "MRY",
    "MPP",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
)


def _attach_special_circuit_kernel(
    kernel: KirinKernel,
    special_circuit_kernel: SquinKernel,
    *,
    num_qubits: int,
) -> KirinKernel:
    """Attach special-prefix source metadata to a generated task kernel."""

    setattr(kernel, "_msd_special_circuit_kernel", special_circuit_kernel)
    setattr(kernel, "_msd_special_circuit_num_qubits", int(num_qubits))
    return kernel


def _build_physical_prefix_source_tsim_circuit(
    special_circuit_kernel: SquinKernel,
    *,
    num_logical_qubits: int,
) -> TsimCircuit:
    """Compile a logical special-prefix source into a physical tsim circuit."""

    from bloqade.lanes.rewrite.squin2stim import RemoveReturn

    @squin.kernel
    def physical_prefix_source():
        q = squin.qalloc(7 * num_logical_qubits)
        inputs = q[6::7]
        special_circuit_kernel(inputs)
        return

    prefix_kernel = physical_prefix_source.similar()
    rewrite.Walk(RemoveReturn()).rewrite(prefix_kernel.code)
    return tsim.Circuit(prefix_kernel)


# TODO: this is for the use case of overriding an existing tsim circuit, and
# overriding the GeminiLogicalSimulatorTask cached attributes. Ideally, we'd add
# this functionality in GeminiLogicalSimulatorTask.
def _set_task_override(
    task: GeminiLogicalSimulatorTask,
    attr: str,
    value: object,
) -> None:
    """Set or shadow a cached Gemini task attribute."""

    try:
        setattr(task, attr, value)
    except AttributeError:
        task.__dict__[attr] = value


# TODO: see above.
def _clear_task_tsim_artifacts(task: GeminiLogicalSimulatorTask) -> None:
    """Clear cached tsim/sampler artifacts after overriding a circuit."""

    for attr in (
        "tsim_circuit",
        "noiseless_tsim_circuit",
        "measurement_sampler",
        "noiseless_measurement_sampler",
        "detector_sampler",
        "noiseless_detector_sampler",
        "detector_error_model",
    ):
        task.__dict__.pop(attr, None)


# TODO: this could be a feature in tsim, to expose the circuit itself w/o
# measurement, or applying an inversion on the circuit ignoring measurement
def _first_nonunitary_instruction_index(circuit: TsimCircuit) -> int:
    """Return the first instruction index that cannot be inverted as unitary."""

    for idx in range(len(circuit)):
        if str(circuit[idx]).startswith(_NONUNITARY_PREFIXES):
            return idx
    return len(circuit)


def _prepend_inverse_tsim_circuit(
    task: GeminiLogicalSimulatorTask,
    circuit_to_invert: TsimCircuit,
) -> None:
    """Prepend the inverse of a unitary prefix circuit to a task."""

    inverse_prefix = circuit_to_invert.without_noise().inverse()
    _override_task_tsim_circuit(
        task,
        inverse_prefix + task.tsim_circuit,
        noiseless_circuit=inverse_prefix + task.noiseless_tsim_circuit,
    )


# TODO: this could be a feature in tsim, to expose the circuit itself w/o
# measurement, or applying an inversion on the circuit ignoring measurement
def _build_compiled_unitary_prefix_circuit(
    task: GeminiLogicalSimulatorTask,
) -> TsimCircuit:
    """Return the initial unitary portion of a compiled task circuit."""

    compiled_circuit = task.tsim_circuit
    return compiled_circuit[: _first_nonunitary_instruction_index(compiled_circuit)]


# TODO: ideally, overriding the tsim circuit could be done in the
# GeminiLogicalSimulatorTask. See above
def _override_task_tsim_circuit(
    task: GeminiLogicalSimulatorTask,
    circuit: TsimCircuit,
    *,
    noiseless_circuit: TsimCircuit | None = None,
) -> None:
    """Replace a task's tsim circuits and clear dependent caches."""

    _clear_task_tsim_artifacts(task)
    _set_task_override(task, "tsim_circuit", circuit)
    if noiseless_circuit is not None:
        _set_task_override(task, "noiseless_tsim_circuit", noiseless_circuit)


def _build_task(
    simulator: GeminiLogicalSimulator,
    kernel: KirinKernel,
    *,
    m2dets: MeasurementMap | None,
    m2obs: MeasurementMap | None,
    append_measurements: bool = True,
) -> DemoTask:
    """Build a demo task from a logical kernel and measurement maps."""

    logical_kernel = kernel.similar()

    task = simulator.task(
        logical_kernel,
        m2dets if append_measurements else None,
        m2obs if append_measurements else None,
    )

    return DemoTask(
        task=task,
        metadata={"logical_kernel": kernel},
    )


def _apply_prefix_prepare_to_task(demo_task: DemoTask) -> None:
    """Apply the prefix-prepare special strategy to one task."""

    kernel = demo_task.metadata.get("logical_kernel")
    if not isinstance(kernel, ir.Method):
        raise ValueError("prefix_prepare special tasks must preserve kernel metadata.")

    special_circuit_kernel = getattr(
        kernel,
        "_msd_special_circuit_kernel",
        None,
    )
    if special_circuit_kernel is None:
        raise ValueError(
            "prefix_prepare special kernels must provide an "
            "_msd_special_circuit_kernel source."
        )
    special_circuit_num_qubits = getattr(
        kernel,
        "_msd_special_circuit_num_qubits",
        None,
    )
    if special_circuit_num_qubits is None:
        raise ValueError(
            "prefix_prepare special kernels must provide an "
            "_msd_special_circuit_num_qubits value."
        )

    prefix_source_circuit = _build_physical_prefix_source_tsim_circuit(
        cast(SquinKernel, special_circuit_kernel),
        num_logical_qubits=int(special_circuit_num_qubits),
    )
    _prepend_inverse_tsim_circuit(
        demo_task.task,
        prefix_source_circuit,
    )


def apply_special_tsim_circuit_strategy(
    task_map: Mapping[str, DemoTask],
    strategy: Literal["prefix_prepare", "compiled_inverse_prefix"] | None,
    *,
    normalize_observable_reference: bool = True,
) -> dict[str, DemoTask]:
    """Apply a special-task circuit-prefix strategy to a task map. This is used to
    make the observable of a circuit deterministic for DEM computation.

    Args:
        task_map: Basis-to-task mapping to transform.
        strategy: Special circuit strategy. ``"prefix_prepare"`` prepends the
            inverse of the special logical source circuit. ``"compiled_inverse_prefix"``
            prepends the inverse of the compiled unitary circuit prefix.
            ``None`` returns a shallow copy of ``task_map``.
        normalize_observable_reference: Whether transformed tasks should rebase
            observables by their deterministic noiseless reference row.

    Returns:
        A new basis-to-task mapping containing the transformed tasks.

    Raises:
        ValueError: If ``strategy`` is not supported or required prefix metadata
            is missing.
    """

    if strategy is None:
        return dict(task_map)
    if strategy not in {"prefix_prepare", "compiled_inverse_prefix"}:
        raise ValueError(
            "special_tsim_circuit_strategy must be 'prefix_prepare', "
            "'compiled_inverse_prefix', or None."
        )

    transformed = dict(task_map)
    for demo_task in transformed.values():
        if strategy == "prefix_prepare":
            _apply_prefix_prepare_to_task(demo_task)
        else:
            _prepend_inverse_tsim_circuit(
                demo_task.task,
                _build_compiled_unitary_prefix_circuit(demo_task.task),
            )
        if normalize_observable_reference:
            demo_task.observable_frame = _ObservableFrame.NOISELESS_REFERENCE_FLIPS
    return transformed


def build_task_map(
    simulator: GeminiLogicalSimulator,
    kernel_map: Mapping[str, KirinKernel],
    *,
    m2dets: MeasurementMap | None,
    m2obs: MeasurementMap | None,
    append_measurements: bool = True,
) -> dict[str, DemoTask]:
    """Build demo tasks for each basis kernel in a kernel map.
    Note that the compilation pipeline will be performed during task construction.

    Args:
        simulator: Gemini logical simulator used to construct tasks.
        kernel_map: Mapping from basis label to logical kernel.
        m2dets: Measurement-to-detector map, or ``None``.
        m2obs: Measurement-to-observable map, or ``None``.
        append_measurements: Whether to pass measurement maps into task
            construction.

    Returns:
        Mapping from basis label to ``DemoTask``.
    """

    return {
        basis: _build_task(
            simulator,
            kernel,
            m2dets=m2dets,
            m2obs=m2obs,
            append_measurements=append_measurements,
        )
        for basis, kernel in kernel_map.items()
    }
