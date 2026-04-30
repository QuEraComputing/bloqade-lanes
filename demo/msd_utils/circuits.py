from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from kirin.dialects import func, ilist

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.device import GeminiLogicalSimulatorTask
from bloqade.gemini.logical.stdlib import default_post_processing
from bloqade.lanes import GeminiLogicalSimulator
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.logical_mvp import (
    append_measurements_and_annotations,
    compile_squin_to_move,
)
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs
from bloqade.lanes.transform import MoveToSquin

from .common import DemoTask, LogicalKernelSpec, ObservableFrame

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


@dataclass(frozen=True)
class NaiveKernelBundle:
    distilled: dict[str, LogicalKernelSpec]
    injected: dict[str, LogicalKernelSpec]


@dataclass(frozen=True)
class DecoderKernelBundle:
    actual: dict[str, LogicalKernelSpec]
    special: dict[str, LogicalKernelSpec]
    injected: dict[str, LogicalKernelSpec]


@dataclass(frozen=True)
class DecoderPrimitiveSet:
    state_injection_circuit: Any
    logical_circuit: Any
    logical_circuit_inverse: Any

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def build_measurement_maps(num_logical_qubits: int) -> tuple[Any, Any]:
    return steane7_m2dets(num_logical_qubits), steane7_m2obs(num_logical_qubits)


def _build_special_prefix_kernel(
    prepare_kernel: Any,
    initializer_kernel: Any,
):
    @squin.kernel
    def prefix(q):
        q0 = q[0:7]
        q1 = q[7:14]
        q2 = q[14:21]
        q3 = q[21:28]
        q4 = q[28:35]
        inputs = [q0[6], q1[6], q2[6], q3[6], q4[6]]

        prepare_kernel(inputs)
        initializer_kernel(0.0, 0.0, 0.0, q0)
        initializer_kernel(0.0, 0.0, 0.0, q1)
        initializer_kernel(0.0, 0.0, 0.0, q2)
        initializer_kernel(0.0, 0.0, 0.0, q3)
        initializer_kernel(0.0, 0.0, 0.0, q4)

    return prefix


def _attach_special_circuit_kernel(
    kernel: Any,
    special_circuit_kernel: Any,
    *,
    num_qubits: int,
) -> Any:
    setattr(kernel, "_msd_special_circuit_kernel", special_circuit_kernel)
    setattr(kernel, "_msd_special_circuit_num_qubits", int(num_qubits))
    return kernel


def _build_inverse_prepare_kernel_from_cirq(
    special_circuit_kernel: Any,
    *,
    num_qubits: int,
    kernel_name: str,
) -> Any:
    import cirq
    from bloqade.cirq_utils import emit_circuit, load_circuit

    @squin.kernel
    def special_circuit_wrapper():
        reg = squin.qalloc(num_qubits)
        special_circuit_kernel(reg)

    circuit = emit_circuit(special_circuit_wrapper, ignore_returns=True)
    inverse_circuit = cirq.inverse(circuit)
    return load_circuit(
        inverse_circuit,
        kernel_name=kernel_name,
        register_as_argument=True,
        register_argument_name="inputs",
    )


def _count_initializer_invokes(
    method: Any,
    *,
    initializer_name: str,
) -> int:
    block = method.code.body.blocks[0]
    count = 0
    for stmt in block.stmts:
        if not isinstance(stmt, func.Invoke):
            continue
        callee = getattr(stmt, "callee", None)
        callee_name = (
            getattr(callee, "sym_name", None) or getattr(callee, "name", None) or ""
        )
        if initializer_name in str(callee_name):
            count += 1
    return count


def _set_task_override(task: GeminiLogicalSimulatorTask, attr: str, value: Any) -> None:
    try:
        setattr(task, attr, value)
    except AttributeError:
        task.__dict__[attr] = value


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


def _first_nonunitary_instruction_index(circuit: Any) -> int:
    for idx in range(len(circuit)):
        if str(circuit[idx]).startswith(NONUNITARY_PREFIXES):
            return idx
    return len(circuit)


def _build_compiled_inverse_prefix_circuit(task: GeminiLogicalSimulatorTask):
    compiled_circuit = task.tsim_circuit
    unitary_prefix = compiled_circuit[
        : _first_nonunitary_instruction_index(compiled_circuit)
    ]
    inverse_prefix = unitary_prefix.without_noise().inverse()
    return inverse_prefix + compiled_circuit


def _override_task_tsim_circuit(
    task: GeminiLogicalSimulatorTask,
    circuit: Any,
) -> None:
    _set_task_override(task, "tsim_circuit", circuit)
    _clear_task_tsim_artifacts(task)
    _set_task_override(task, "tsim_circuit", circuit)


def _splice_noiseless_prefix_into_compiled_kernel(
    compiled_kernel: Any,
    prefix_kernel: Any,
    *,
    initializer_name: str,
):
    mt = compiled_kernel.similar()
    block = mt.code.body.blocks[0]
    stmts = list(block.stmts)

    qubit_new_stmts = [
        stmt for stmt in stmts if "qubit.new" in stmt.name or stmt.name == "new"
    ]
    if len(qubit_new_stmts) < 35:
        raise RuntimeError(
            f"Expected at least 35 qubit allocations, got {len(qubit_new_stmts)}"
        )

    init_invokes = []
    for stmt in stmts:
        if not isinstance(stmt, func.Invoke):
            continue
        callee = getattr(stmt, "callee", None)
        callee_name = (
            getattr(callee, "sym_name", None) or getattr(callee, "name", None) or ""
        )
        if initializer_name in str(callee_name):
            init_invokes.append(stmt)

    if len(init_invokes) != 5:
        raise RuntimeError(f"Expected 5 initializer invokes, got {len(init_invokes)}")

    anchor = init_invokes[0]
    full_reg = ilist.New(tuple(stmt.result for stmt in qubit_new_stmts[:35]))
    full_reg.insert_before(anchor)
    func.Invoke((full_reg.result,), callee=prefix_kernel).insert_before(anchor)

    for stmt in reversed(init_invokes):
        stmt.delete(safe=False)

    return mt


def _apply_special_state_prefix(
    compiled_kernel: Any,
    *,
    prepare_kernel: Any,
    initializer_kernel: Any,
    initializer_name: str,
) -> Any:
    prefix_kernel = _build_special_prefix_kernel(prepare_kernel, initializer_kernel)
    return _splice_noiseless_prefix_into_compiled_kernel(
        compiled_kernel,
        prefix_kernel,
        initializer_name=initializer_name,
    )


def _ensure_kernel_spec(kernel_like: LogicalKernelSpec | Any) -> LogicalKernelSpec:
    if isinstance(kernel_like, LogicalKernelSpec):
        return kernel_like
    return LogicalKernelSpec(kernel=kernel_like)


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

    @squin.kernel
    def msd_inverse(reg):
        squin.broadcast.sqrt_x(reg)
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.sqrt_x(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.broadcast.sqrt_y_adj(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_x_adj(ilist.IList([reg[0], reg[1], reg[4]]))

    return DecoderPrimitiveSet(
        state_injection_circuit=msd_magic_prep,
        logical_circuit=msd_forward,
        logical_circuit_inverse=msd_inverse,
    )


def _build_tomography_primitives(*, output_qubit: int):
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

    @squin.kernel
    def tomography_x_inv(reg):
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_y_inv(reg):
        squin.h(reg[output_qubit])
        squin.sqrt_z(reg[output_qubit])

    @squin.kernel
    def tomography_z_inv(reg):
        return

    return {
        "tomography_x": tomography_x,
        "tomography_y": tomography_y,
        "tomography_z": tomography_z,
        "tomography_x_inv": tomography_x_inv,
        "tomography_y_inv": tomography_y_inv,
        "tomography_z_inv": tomography_z_inv,
    }


def _require_primitive_keys(
    primitives: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    builder_name: str,
) -> None:
    missing = [key for key in keys if key not in primitives]
    if missing:
        raise ValueError(
            f"{builder_name} must return keys {keys}; missing {tuple(missing)}."
        )


def _coerce_decoder_primitive_set(
    primitive_set: DecoderPrimitiveSet | Mapping[str, Any],
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
                "logical_circuit_inverse",
            ),
            builder_name=builder_name,
        )
        return DecoderPrimitiveSet(
            state_injection_circuit=primitive_set["state_injection_circuit"],
            logical_circuit=primitive_set["logical_circuit"],
            logical_circuit_inverse=primitive_set["logical_circuit_inverse"],
        )
    raise TypeError(
        f"{builder_name} must return a DecoderPrimitiveSet or mapping, got "
        f"{type(primitive_set).__name__}."
    )


def build_naive_kernel_bundle(
    theta: float,
    phi: float,
    lam: float,
    *,
    output_qubit: int = 0,
) -> NaiveKernelBundle:
    msd_primitives = _build_msd_primitives(theta, phi, lam)
    tomography_primitives = _build_tomography_primitives(output_qubit=output_qubit)
    state_injection_circuit = msd_primitives["state_injection_circuit"]
    logical_circuit = msd_primitives["logical_circuit"]
    tomography_x = tomography_primitives["tomography_x"]
    tomography_y = tomography_primitives["tomography_y"]
    tomography_z = tomography_primitives["tomography_z"]

    @gemini_logical.kernel(aggressive_unroll=True)
    def distilled_x():
        reg = qubit.qalloc(5)
        state_injection_circuit(reg)
        logical_circuit(reg)
        tomography_x(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def distilled_y():
        reg = qubit.qalloc(5)
        state_injection_circuit(reg)
        logical_circuit(reg)
        tomography_y(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def distilled_z():
        reg = qubit.qalloc(5)
        state_injection_circuit(reg)
        logical_circuit(reg)
        tomography_z(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_x():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_x(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_y():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_y(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_z():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_z(reg)
        return

    return NaiveKernelBundle(
        distilled={
            "X": LogicalKernelSpec(kernel=distilled_x),
            "Y": LogicalKernelSpec(kernel=distilled_y),
            "Z": LogicalKernelSpec(kernel=distilled_z),
        },
        injected={
            "X": LogicalKernelSpec(kernel=injected_x),
            "Y": LogicalKernelSpec(kernel=injected_y),
            "Z": LogicalKernelSpec(kernel=injected_z),
        },
    )


def build_decoder_kernel_bundle(
    *primitive_args: float,
    num_logical_qubits: int = 5,
    output_qubit: int = 0,
    build_primitives: Callable[..., DecoderPrimitiveSet | Mapping[str, Any]] = (
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
    tomography_x = tomography_primitives["tomography_x"]
    tomography_y = tomography_primitives["tomography_y"]
    tomography_z = tomography_primitives["tomography_z"]

    @squin.kernel
    def special_circuit_x(reg):
        logical_circuit(reg)
        tomography_x(reg)

    @squin.kernel
    def special_circuit_y(reg):
        logical_circuit(reg)
        tomography_y(reg)

    @squin.kernel
    def special_circuit_z(reg):
        logical_circuit(reg)
        tomography_z(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_actual_x():
        reg = qubit.qalloc(num_logical_qubits)
        state_injection_circuit(reg)
        logical_circuit(reg)
        tomography_x(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_actual_y():
        reg = qubit.qalloc(num_logical_qubits)
        state_injection_circuit(reg)
        logical_circuit(reg)
        tomography_y(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_actual_z():
        reg = qubit.qalloc(num_logical_qubits)
        state_injection_circuit(reg)
        logical_circuit(reg)
        tomography_z(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_special_x():
        reg = qubit.qalloc(num_logical_qubits)
        logical_circuit(reg)
        tomography_x(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_special_y():
        reg = qubit.qalloc(num_logical_qubits)
        logical_circuit(reg)
        tomography_y(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_special_z():
        reg = qubit.qalloc(num_logical_qubits)
        logical_circuit(reg)
        tomography_z(reg)
        return default_post_processing(reg)

    actual = {
        "X": LogicalKernelSpec(kernel=msd_actual_x),
        "Y": LogicalKernelSpec(kernel=msd_actual_y),
        "Z": LogicalKernelSpec(kernel=msd_actual_z),
    }
    if special_kernel_strategy == "prefix_prepare":
        special = {
            "X": LogicalKernelSpec(
                kernel=_attach_special_circuit_kernel(
                    msd_special_x,
                    special_circuit_x,
                    num_qubits=num_logical_qubits,
                ),
                special_tsim_circuit_strategy="prefix_prepare",
                observable_frame=ObservableFrame.NOISELESS_REFERENCE_FLIPS,
            ),
            "Y": LogicalKernelSpec(
                kernel=_attach_special_circuit_kernel(
                    msd_special_y,
                    special_circuit_y,
                    num_qubits=num_logical_qubits,
                ),
                special_tsim_circuit_strategy="prefix_prepare",
                observable_frame=ObservableFrame.NOISELESS_REFERENCE_FLIPS,
            ),
            "Z": LogicalKernelSpec(
                kernel=_attach_special_circuit_kernel(
                    msd_special_z,
                    special_circuit_z,
                    num_qubits=num_logical_qubits,
                ),
                special_tsim_circuit_strategy="prefix_prepare",
                observable_frame=ObservableFrame.NOISELESS_REFERENCE_FLIPS,
            ),
        }
    else:
        special = {
            "X": LogicalKernelSpec(
                kernel=msd_actual_x,
                special_tsim_circuit_strategy="compiled_inverse_prefix",
                observable_frame=ObservableFrame.NOISELESS_REFERENCE_FLIPS,
            ),
            "Y": LogicalKernelSpec(
                kernel=msd_actual_y,
                special_tsim_circuit_strategy="compiled_inverse_prefix",
                observable_frame=ObservableFrame.NOISELESS_REFERENCE_FLIPS,
            ),
            "Z": LogicalKernelSpec(
                kernel=msd_actual_z,
                special_tsim_circuit_strategy="compiled_inverse_prefix",
                observable_frame=ObservableFrame.NOISELESS_REFERENCE_FLIPS,
            ),
        }

    resolved_injected_prep_args = injected_prep_args
    if resolved_injected_prep_args is None and len(primitive_args) == 3:
        resolved_injected_prep_args = (
            float(primitive_args[0]),
            float(primitive_args[1]),
            float(primitive_args[2]),
        )

    injected: dict[str, LogicalKernelSpec] = {}
    if resolved_injected_prep_args is not None:
        theta, phi, lam = resolved_injected_prep_args

        @gemini_logical.kernel(aggressive_unroll=True)
        def injected_x():
            reg = qubit.qalloc(1)
            squin.u3(theta, phi, lam, reg[0])
            tomography_x(reg)
            return default_post_processing(reg)

        @gemini_logical.kernel(aggressive_unroll=True)
        def injected_y():
            reg = qubit.qalloc(1)
            squin.u3(theta, phi, lam, reg[0])
            tomography_y(reg)
            return default_post_processing(reg)

        @gemini_logical.kernel(aggressive_unroll=True)
        def injected_z():
            reg = qubit.qalloc(1)
            squin.u3(theta, phi, lam, reg[0])
            tomography_z(reg)
            return default_post_processing(reg)

        injected = {
            "X": LogicalKernelSpec(kernel=injected_x),
            "Y": LogicalKernelSpec(kernel=injected_y),
            "Z": LogicalKernelSpec(kernel=injected_z),
        }

    return DecoderKernelBundle(
        actual=actual,
        special=special,
        injected=injected,
    )


def build_injected_decoder_kernel_map(
    *,
    output_qubit: int = 0,
) -> dict[str, LogicalKernelSpec]:
    h_theta = 0.5 * math.pi
    h_phi = 0.0
    hs_theta = 0.5 * math.pi
    hs_phi = -0.5 * math.pi
    lam = 0.0
    tomography_primitives = _build_tomography_primitives(output_qubit=output_qubit)
    tomography_x = tomography_primitives["tomography_x"]
    tomography_y = tomography_primitives["tomography_y"]
    tomography_z = tomography_primitives["tomography_z"]

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_decoder_x():
        reg = qubit.qalloc(1)
        squin.u3(h_theta, h_phi, lam, reg[0])
        tomography_x(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_decoder_y():
        reg = qubit.qalloc(1)
        squin.u3(hs_theta, hs_phi, lam, reg[0])
        tomography_y(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_decoder_z():
        reg = qubit.qalloc(1)
        squin.u3(0.0, 0.0, 0.0, reg[0])
        tomography_z(reg)
        return default_post_processing(reg)

    return {
        "X": LogicalKernelSpec(kernel=injected_decoder_x),
        "Y": LogicalKernelSpec(kernel=injected_decoder_y),
        "Z": LogicalKernelSpec(kernel=injected_decoder_z),
    }


def make_noisy_steane7_initializer(simulator: GeminiLogicalSimulator):
    local_r_noise = simulator.noise_model.local_r_noise
    local_rz_noise = simulator.noise_model.local_rz_noise
    cz_paired_noise = simulator.noise_model.cz_paired_noise
    cz_unpaired_noise = simulator.noise_model.cz_unpaired_noise

    x_axis = 0.0
    y_axis = 0.25
    half_turn = 0.5
    quarter_turn = 0.25

    @squin.kernel
    def noisy_steane7_initialize(theta, phi, lam, qubits):
        evens = qubits[::2]
        odds = qubits[1::2]

        squin.u3(theta, phi, lam, qubits[6])
        local_r_noise(ilist.IList([qubits[6]]), x_axis, half_turn)

        first6 = qubits[:6]
        squin.broadcast.sqrt_y_adj(first6)
        local_r_noise(first6, y_axis, quarter_turn)

        targets1 = evens[1:]
        squin.broadcast.cz(odds, targets1)
        cz_paired_noise(odds, targets1)
        cz_unpaired_noise(ilist.IList([qubits[0]]))

        squin.sqrt_y(qubits[6])
        local_r_noise(ilist.IList([qubits[6]]), y_axis, quarter_turn)

        controls2 = evens[:-1]
        targets2 = ilist.IList([qubits[3], qubits[5], qubits[6]])
        squin.broadcast.cz(controls2, targets2)
        cz_paired_noise(controls2, targets2)
        cz_unpaired_noise(ilist.IList([qubits[1]]))

        tail = qubits[2:]
        squin.broadcast.sqrt_y(tail)
        local_r_noise(tail, y_axis, quarter_turn)

        controls3 = evens[:-1]
        targets3 = odds
        squin.broadcast.cz(controls3, targets3)
        cz_paired_noise(controls3, targets3)
        cz_unpaired_noise(ilist.IList([qubits[6]]))

        subset = ilist.IList([qubits[1], qubits[2], qubits[4]])
        squin.broadcast.sqrt_y(subset)
        local_r_noise(subset, y_axis, quarter_turn)

        squin.x(qubits[3])
        local_r_noise(ilist.IList([qubits[3]]), x_axis, half_turn)

        z_subset = ilist.IList([qubits[1], qubits[5]])
        squin.broadcast.z(z_subset)
        local_rz_noise(z_subset, half_turn)

    return noisy_steane7_initialize


# TODO, 4/17 10:30 AM: continue reading from here to think about how to integrate Jing's code.
def build_task(
    simulator: GeminiLogicalSimulator,
    kernel_spec: LogicalKernelSpec | Any,
    *,
    m2dets: Any,
    m2obs: Any,
    noisy_initializer: Any | None = None,
    append_measurements: bool = True,
    physical_hypercube_dims: int = 4,
    transversal_rewrite: bool = True,
) -> DemoTask:
    from bloqade.lanes.arch.gemini.logical import steane7_initialize

    spec = _ensure_kernel_spec(kernel_spec)
    logical_kernel = spec.kernel.similar()
    if append_measurements:
        append_measurements_and_annotations(logical_kernel, m2dets, m2obs)

    physical_arch_spec = generate_arch_hypercube(physical_hypercube_dims)
    physical_move_kernel = compile_squin_to_move(
        logical_kernel,
        transversal_rewrite=transversal_rewrite,
    )
    post_processing = atom.AtomInterpreter(
        physical_move_kernel.dialects,
        arch_spec=physical_arch_spec,
    ).get_post_processing(physical_move_kernel)

    task = GeminiLogicalSimulatorTask(
        logical_kernel,
        simulator.noise_model,
        physical_arch_spec,
        physical_move_kernel,
        post_processing,
    )

    if noisy_initializer is not None:
        _set_task_override(
            task,
            "physical_squin_kernel",
            MoveToSquin(
                physical_arch_spec,
                logical_initialization=noisy_initializer,
                noise_model=simulator.noise_model,
            ).emit(physical_move_kernel),
        )

    if spec.special_tsim_circuit_strategy == "prefix_prepare":
        special_circuit_kernel = getattr(
            spec.kernel,
            "_msd_special_circuit_kernel",
            None,
        )
        if special_circuit_kernel is None:
            raise ValueError(
                "prefix_prepare special kernels must provide an "
                "_msd_special_circuit_kernel source."
            )
        special_circuit_num_qubits = getattr(
            spec.kernel,
            "_msd_special_circuit_num_qubits",
            None,
        )
        if special_circuit_num_qubits is None:
            raise ValueError(
                "prefix_prepare special kernels must provide an "
                "_msd_special_circuit_num_qubits value."
            )
        special_prepare_kernel = _build_inverse_prepare_kernel_from_cirq(
            special_circuit_kernel,
            num_qubits=int(special_circuit_num_qubits),
            kernel_name=(
                f"{getattr(spec.kernel, 'sym_name', 'special')}_prepare_inverse"
            ),
        )
        initializer_kernel = noisy_initializer or steane7_initialize
        initializer_name = (
            getattr(initializer_kernel, "sym_name", None)
            or getattr(initializer_kernel, "name", None)
            or initializer_kernel.__name__
        )

        compiled_kernel = getattr(task, "physical_squin_kernel")
        _set_task_override(
            task,
            "physical_squin_kernel",
            _apply_special_state_prefix(
                compiled_kernel,
                prepare_kernel=special_prepare_kernel,
                initializer_kernel=initializer_kernel,
                initializer_name=str(initializer_name),
            ),
        )
    elif spec.special_tsim_circuit_strategy == "compiled_inverse_prefix":
        _override_task_tsim_circuit(task, _build_compiled_inverse_prefix_circuit(task))
    elif spec.special_tsim_circuit_strategy is not None:
        raise ValueError(
            "Unknown special_tsim_circuit_strategy: "
            f"{spec.special_tsim_circuit_strategy}"
        )

    return DemoTask(
        task=task,
        observable_frame=spec.observable_frame,
        metadata={"logical_kernel_spec": spec},
    )


def build_task_map(
    simulator: GeminiLogicalSimulator,
    kernel_map: Mapping[str, LogicalKernelSpec | Any],
    *,
    m2dets: Any,
    m2obs: Any,
    noisy_initializer: Any | None = None,
    append_measurements: bool = True,
    physical_hypercube_dims: int = 4,
    transversal_rewrite: bool = True,
) -> dict[str, DemoTask]:
    return {
        basis: build_task(
            simulator,
            kernel,
            m2dets=m2dets,
            m2obs=m2obs,
            noisy_initializer=noisy_initializer,
            append_measurements=append_measurements,
            physical_hypercube_dims=physical_hypercube_dims,
            transversal_rewrite=transversal_rewrite,
        )
        for basis, kernel in kernel_map.items()
    }
