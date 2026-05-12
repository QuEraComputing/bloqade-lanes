from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, TypeAlias, cast

import tsim as tsim_backend
from kirin import ir, rewrite
from kirin.dialects import ilist

from bloqade import qubit, squin, tsim
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.device import GeminiLogicalSimulatorTask
from bloqade.gemini.logical.stdlib import default_post_processing
from bloqade.lanes import GeminiLogicalSimulator
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs

from .common import DemoTask, ObservableFrame

# TODO: Apparently we can't type check for kernels written in specific dialects, so all we can check is that something is a Kirin kernel I suppose,
# which is ir.Method[..., Any]. And we are just "visually enforcing" the types through KirinKernel and SquinKernel. Think about a better way to deal with types here.
KirinKernel: TypeAlias = ir.Method[..., Any]
SquinKernel: TypeAlias = KirinKernel
TsimCircuit: TypeAlias = tsim_backend.Circuit
MeasurementMap: TypeAlias = list[list[int]]


# REFACTOR: this should be an internal constant for stdlibs
NONUNITARY_PREFIXES = (
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


# REFACTOR: This should be a more "application-specific" datatype; I don't think this should live in the standard libraries.
@dataclass(frozen=True)
class DecoderKernelBundle:
    actual: dict[str, KirinKernel]
    special: dict[str, KirinKernel]
    injected: dict[str, KirinKernel]


# REFACTOR: This should be a more "application-specific" datatype; I don't think this should live in the standard libraries.
@dataclass(frozen=True)
class DecoderPrimitiveSet:
    state_injection_circuit: SquinKernel
    logical_circuit: SquinKernel

    def __getitem__(self, key: str) -> SquinKernel:
        return getattr(self, key)


# REFACTOR: this should be a standard library function.
def build_measurement_maps(
    num_logical_qubits: int,
) -> tuple[MeasurementMap, MeasurementMap]:
    return steane7_m2dets(num_logical_qubits), steane7_m2obs(num_logical_qubits)


# REFACTOR: this should be an internal function for more "application-level" functions.
def _attach_special_circuit_kernel(
    kernel: KirinKernel,
    special_circuit_kernel: SquinKernel,
    *,
    num_qubits: int,
) -> KirinKernel:
    setattr(kernel, "_msd_special_circuit_kernel", special_circuit_kernel)
    setattr(kernel, "_msd_special_circuit_num_qubits", int(num_qubits))
    return kernel


# REFACTOR: this should be an internal function for an application-level function.
def _build_physical_prefix_source_tsim_circuit(
    special_circuit_kernel: SquinKernel,
    *,
    num_logical_qubits: int,
) -> TsimCircuit:
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


# REFACTOR: this should be an internal function for an application-level function.
# TODO: this is for the use case of overriding an existing tsim circuit, and overriding the GeminiLogicalSimulatorTask
# cached attributes. Ideally, we'd add this functionality in GeminiLogicalSimulatorTask.
def _set_task_override(
    task: GeminiLogicalSimulatorTask,
    attr: str,
    value: object,
) -> None:
    try:
        setattr(task, attr, value)
    except AttributeError:
        task.__dict__[attr] = value


# REFACTOR: this should be an internal function for an application-level function.
# TODO: see above.
def _clear_task_tsim_artifacts(task: GeminiLogicalSimulatorTask) -> None:
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


# REFACTOR: this should be an internal function for an application-level function.
# TODO: this could be a feature in tsim, to expose the circuit itself w/o measurement, or applying an inversion on the circuit ignoring measurement
def _first_nonunitary_instruction_index(circuit: TsimCircuit) -> int:
    for idx in range(len(circuit)):
        if str(circuit[idx]).startswith(NONUNITARY_PREFIXES):
            return idx
    return len(circuit)


# REFACTOR: this should be an internal function for an application-level function.
def _prepend_inverse_tsim_circuit(
    task: GeminiLogicalSimulatorTask,
    circuit_to_invert: TsimCircuit,
) -> None:
    inverse_prefix = circuit_to_invert.without_noise().inverse()
    _override_task_tsim_circuit(
        task,
        inverse_prefix + task.tsim_circuit,
        noiseless_circuit=inverse_prefix + task.noiseless_tsim_circuit,
    )


# TODO: this could be a feature in tsim, to expose the circuit itself w/o measurement, or applying an inversion on the circuit ignoring measurement
def _build_compiled_unitary_prefix_circuit(
    task: GeminiLogicalSimulatorTask,
) -> TsimCircuit:
    compiled_circuit = task.tsim_circuit
    return compiled_circuit[: _first_nonunitary_instruction_index(compiled_circuit)]


# TODO: ideally, overriding the tsim circuit could be done in the GeminiLogicalSimulatorTask. See above
def _override_task_tsim_circuit(
    task: GeminiLogicalSimulatorTask,
    circuit: TsimCircuit,
    *,
    noiseless_circuit: TsimCircuit | None = None,
) -> None:
    _clear_task_tsim_artifacts(task)
    _set_task_override(task, "tsim_circuit", circuit)
    if noiseless_circuit is not None:
        _set_task_override(task, "noiseless_tsim_circuit", noiseless_circuit)


# NOTE: this is basically what the user would "instantiate" for this specific MSD experiment
def _build_msd_primitives(
    theta: float,
    phi: float,
    lam: float,
) -> DecoderPrimitiveSet:
    @squin.kernel
    def msd_magic_prep(reg):
        squin.broadcast.u3(theta, phi, lam, reg)

    @squin.kernel
    def msd_forward(reg):
        squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.sqrt_x_adj(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.broadcast.sqrt_x_adj(reg)

    return DecoderPrimitiveSet(
        state_injection_circuit=msd_magic_prep,
        logical_circuit=msd_forward,
    )


def _build_tomography_primitives(*, output_qubit: int) -> dict[str, SquinKernel]:
    @squin.kernel
    def tomography_x(reg):
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_y(reg):
        squin.sqrt_z_adj(reg[output_qubit])
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_z(reg):
        return

    return {
        "tomography_x": tomography_x,
        "tomography_y": tomography_y,
        "tomography_z": tomography_z,
    }


@squin.kernel
def _squin_return_none(reg):
    return


def produce_tomography_kernels(
    num_qubits: int,
    logical_kernel: KirinKernel,
    tomography_kernels: Mapping[str, SquinKernel],
    return_val_fn: KirinKernel,
    kernel_name: str,
    *,
    supply_reg: bool = True,
) -> Mapping[str, KirinKernel]:
    def make_kernel(tomog_kernel: SquinKernel, generated_name: str) -> KirinKernel:
        def inner_tomog_kernel(reg):
            logical_kernel(reg)
            tomog_kernel(reg)

        inner_tomog_kernel.__name__ = generated_name
        inner_tomog_kernel.__qualname__ = generated_name
        inner_kernel = squin.kernel(inner_tomog_kernel)

        if not supply_reg:
            return inner_kernel

        def alloc_kernel():
            reg = qubit.qalloc(num_qubits)
            inner_kernel(reg)
            return return_val_fn(reg)

        alloc_kernel.__name__ = generated_name
        alloc_kernel.__qualname__ = generated_name
        return gemini_logical.kernel(aggressive_unroll=True)(alloc_kernel)

    return {
        f"{kernel_name}_{tomog_kernel_key.split('_')[-1]}": make_kernel(
            tomog_kernel,
            f"{kernel_name}_{tomog_kernel_key.split('_')[-1]}",
        )
        for tomog_kernel_key, tomog_kernel in tomography_kernels.items()
    }


# This is to give us a dictionary of form {"X": ..., "Y": ..., "Z": ...} for downstream consumption
def _kernels_by_tomography_basis(
    kernels: Mapping[str, KirinKernel],
) -> dict[str, KirinKernel]:
    return {
        kernel_name.split("_")[-1].upper(): kernel
        for kernel_name, kernel in kernels.items()
    }


# NOTE: this is to basically enforce typing at runtime for Python.. in part because we don't have compile-time checks
def _require_primitive_keys(
    primitives: Mapping[str, SquinKernel],
    *,
    keys: tuple[str, ...],
    builder_name: str,
) -> None:
    missing = [key for key in keys if key not in primitives]
    if missing:
        raise ValueError(
            f"{builder_name} must return keys {keys}; missing {tuple(missing)}."
        )


# NOTE: this is to basically enforce typing at runtime for Python.. in part because we don't have compile-time checks
def _coerce_decoder_primitive_set(
    primitive_set: DecoderPrimitiveSet | Mapping[str, SquinKernel],
    *,
    builder_name: str,
) -> DecoderPrimitiveSet:
    if isinstance(primitive_set, DecoderPrimitiveSet):
        return primitive_set
    if isinstance(primitive_set, Mapping):
        _require_primitive_keys(
            primitive_set,
            keys=(
                "state_injection_circuit",
                "logical_circuit",
            ),
            builder_name=builder_name,
        )
        return DecoderPrimitiveSet(
            state_injection_circuit=primitive_set["state_injection_circuit"],
            logical_circuit=primitive_set["logical_circuit"],
        )
    raise TypeError(
        f"{builder_name} must return a DecoderPrimitiveSet or mapping, got "
        f"{type(primitive_set).__name__}."
    )


def build_decoder_kernel_bundle(
    *primitive_args: float,
    # TODO: get rid of logical qubits argument here?
    num_logical_qubits: int = 5,
    output_qubit: int = 0,
    build_primitives: Callable[..., DecoderPrimitiveSet | Mapping[str, SquinKernel]] = (
        _build_msd_primitives
    ),
    injected_prep_args: tuple[float, float, float] | None = None,
    special_kernel_strategy: Literal[
        "prefix_prepare", "compiled_inverse_prefix"
    ] = "prefix_prepare",
) -> DecoderKernelBundle:
    if special_kernel_strategy not in {"prefix_prepare", "compiled_inverse_prefix"}:
        raise ValueError(
            "special_kernel_strategy must be 'prefix_prepare' or "
            "'compiled_inverse_prefix'."
        )

    primitive_set = _coerce_decoder_primitive_set(
        build_primitives(*primitive_args),
        builder_name=getattr(build_primitives, "__name__", "build_primitives"),
    )
    tomography_primitives = _build_tomography_primitives(output_qubit=output_qubit)
    state_injection_circuit = primitive_set.state_injection_circuit
    logical_circuit = primitive_set.logical_circuit

    @squin.kernel
    def actual_logical_kernel(reg):
        state_injection_circuit(reg)
        logical_circuit(reg)

    @squin.kernel
    def special_logical_kernel(reg):
        logical_circuit(reg)

    actual_kernels = produce_tomography_kernels(
        num_logical_qubits,
        actual_logical_kernel,
        tomography_primitives,
        default_post_processing,
        "msd_actual",
    )
    special_task_kernels = produce_tomography_kernels(
        num_logical_qubits,
        special_logical_kernel,
        tomography_primitives,
        default_post_processing,
        "msd_special",
    )
    special_circuit_sources = _kernels_by_tomography_basis(
        produce_tomography_kernels(
            num_logical_qubits,
            special_logical_kernel,
            tomography_primitives,
            _squin_return_none,
            "msd_special_circuit",
            supply_reg=False,
        )
    )

    actual = _kernels_by_tomography_basis(actual_kernels)
    if special_kernel_strategy == "prefix_prepare":
        # TODO: think about a cleaner way to pass down this information? Do I have to pass down this "special_kernel_{x, y, z}"?
        special = {
            basis: _attach_special_circuit_kernel(
                kernel,
                special_circuit_sources[basis],
                num_qubits=num_logical_qubits,
            )
            for basis, kernel in _kernels_by_tomography_basis(
                special_task_kernels
            ).items()
        }
    else:
        special = dict(actual)

    resolved_injected_prep_args = injected_prep_args
    if resolved_injected_prep_args is None and len(primitive_args) == 3:
        resolved_injected_prep_args = (
            float(primitive_args[0]),
            float(primitive_args[1]),
            float(primitive_args[2]),
        )

    injected: dict[str, KirinKernel] = {}
    if resolved_injected_prep_args is not None:
        theta, phi, lam = resolved_injected_prep_args

        @squin.kernel
        def injected_logical_kernel(reg):
            squin.u3(theta, phi, lam, reg[0])

        injected_kernels = produce_tomography_kernels(
            1,
            injected_logical_kernel,
            tomography_primitives,
            default_post_processing,
            "injected",
        )

        injected = _kernels_by_tomography_basis(injected_kernels)

    return DecoderKernelBundle(
        actual=actual,
        special=special,
        injected=injected,
    )


def build_injected_decoder_kernel_map(
    *,
    output_qubit: int = 0,
) -> dict[str, KirinKernel]:
    h_theta = 0.5 * math.pi
    h_phi = 0.0
    hs_theta = 0.5 * math.pi
    hs_phi = -0.5 * math.pi
    lam = 0.0
    tomography_primitives = _build_tomography_primitives(output_qubit=output_qubit)

    @squin.kernel
    def injected_decoder_x_logical_kernel(reg):
        squin.u3(h_theta, h_phi, lam, reg[0])

    @squin.kernel
    def injected_decoder_y_logical_kernel(reg):
        squin.u3(hs_theta, hs_phi, lam, reg[0])

    @squin.kernel
    def injected_decoder_z_logical_kernel(reg):
        squin.u3(0.0, 0.0, 0.0, reg[0])

    return {
        **_kernels_by_tomography_basis(
            produce_tomography_kernels(
                1,
                injected_decoder_x_logical_kernel,
                {"tomography_x": tomography_primitives["tomography_x"]},
                default_post_processing,
                "injected_decoder",
            )
        ),
        **_kernels_by_tomography_basis(
            produce_tomography_kernels(
                1,
                injected_decoder_y_logical_kernel,
                {"tomography_y": tomography_primitives["tomography_y"]},
                default_post_processing,
                "injected_decoder",
            )
        ),
        **_kernels_by_tomography_basis(
            produce_tomography_kernels(
                1,
                injected_decoder_z_logical_kernel,
                {"tomography_z": tomography_primitives["tomography_z"]},
                default_post_processing,
                "injected_decoder",
            )
        ),
    }


def build_task(
    simulator: GeminiLogicalSimulator,
    kernel: KirinKernel,
    *,
    m2dets: MeasurementMap | None,
    m2obs: MeasurementMap | None,
    append_measurements: bool = True,
) -> DemoTask:
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
            demo_task.observable_frame = ObservableFrame.NOISELESS_REFERENCE_FLIPS
    return transformed


def build_task_map(
    simulator: GeminiLogicalSimulator,
    kernel_map: Mapping[str, KirinKernel],
    *,
    m2dets: MeasurementMap | None,
    m2obs: MeasurementMap | None,
    append_measurements: bool = True,
) -> dict[str, DemoTask]:
    return {
        basis: build_task(
            simulator,
            kernel,
            m2dets=m2dets,
            m2obs=m2obs,
            append_measurements=append_measurements,
        )
        for basis, kernel in kernel_map.items()
    }
