#!/usr/bin/env python3
"""Convert QASM -> Cirq -> parallelized decorator-style SQUIN kernel."""

from __future__ import annotations

import argparse
import importlib.util
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import cirq
import numpy as np
from bloqade.cirq_utils.parallelize import parallelize
from cirq.contrib.qasm_import import circuit_from_qasm
from cirq.linalg.decompositions import deconstruct_single_qubit_matrix_into_angles

U3_RE = re.compile(
    r"^\s*squin\.u3\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*q\[(\d+)\]\s*\)\s*$"
)
CZ_RE = re.compile(r"^\s*squin\.cz\(\s*q\[(\d+)\]\s*,\s*q\[(\d+)\]\s*\)\s*$")
BARRIER_RE = re.compile(r"^\s*barrier(?:\s+[^;]+)?\s*;\s*$", re.IGNORECASE)


@dataclass
class U3Op:
    a: str
    b: str
    c: str
    q: int


@dataclass
class CZOp:
    q0: int
    q1: int


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _qubit_sort_key(qubit: cirq.Qid) -> tuple[int, int, int, str]:
    if isinstance(qubit, cirq.LineQubit):
        return (0, qubit.x, 0, "")
    if isinstance(qubit, cirq.GridQubit):
        return (1, qubit.row, qubit.col, "")
    return (2, 0, 0, str(qubit))


def _normalize_kernel_name(name: str) -> str:
    normalized = re.sub(r"\W+", "_", name).strip("_")
    if not normalized:
        normalized = "kernel"
    if normalized[0].isdigit():
        normalized = f"k_{normalized}"
    return normalized


def _strip_terminal_measurements(circuit: cirq.Circuit) -> tuple[cirq.Circuit, int]:
    """Drop measurements that are terminal on each measured qubit."""
    moments = list(circuit)
    last_non_measure_moment: dict[cirq.Qid, int] = {}

    for idx, moment in enumerate(moments):
        for op in moment.operations:
            if cirq.is_measurement(op):
                continue
            for qubit in op.qubits:
                last_non_measure_moment[qubit] = idx

    measurement_count = 0
    kept_moments: list[cirq.Moment] = []

    for idx, moment in enumerate(moments):
        kept_ops: list[cirq.Operation] = []
        for op in moment.operations:
            if not cirq.is_measurement(op):
                kept_ops.append(op)
                continue

            for qubit in op.qubits:
                if idx < last_non_measure_moment.get(qubit, -1):
                    raise ValueError(
                        "Found non-terminal measurement on a qubit with later unitary "
                        "operations, which cannot be converted to a SQUIN kernel."
                    )
            measurement_count += 1

        if kept_ops:
            kept_moments.append(cirq.Moment(kept_ops))

    return cirq.Circuit(kept_moments), measurement_count


def _to_cz_native_cirq(circuit: cirq.Circuit) -> cirq.Circuit:
    """Rewrite a circuit into CZ + single-qubit native operations."""
    return cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())


def _strip_qasm_barriers(qasm_text: str) -> tuple[str, int]:
    """Remove OpenQASM barrier statements before Cirq parsing."""
    kept_lines: list[str] = []
    dropped = 0
    for line in qasm_text.splitlines():
        stripped = line.strip()
        if BARRIER_RE.match(stripped):
            dropped += 1
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines) + "\n", dropped


def _load_qasm_circuit(qasm_text: str) -> tuple[cirq.Circuit, int]:
    """Load OpenQASM with Cirq's native parser and compatibility fallback."""
    from_qasm = getattr(cirq.Circuit, "from_qasm", None)
    if callable(from_qasm):
        try:
            parser = cast(Callable[[str], cirq.Circuit], from_qasm)
            return parser(qasm_text), 0
        except Exception:
            pass

    qasm_without_barriers, dropped = _strip_qasm_barriers(qasm_text)
    try:
        return circuit_from_qasm(qasm_without_barriers), dropped
    except Exception as fallback_error:
        raise ValueError(
            "Unable to parse OpenQASM with either cirq.Circuit.from_qasm or "
            "cirq.contrib.qasm_import.circuit_from_qasm."
        ) from fallback_error


def circuit_to_squin_decorator_source(circuit: cirq.Circuit, kernel_name: str) -> str:
    """Render a Cirq circuit as decorator-style SQUIN Python source."""
    qubits = sorted(circuit.all_qubits(), key=_qubit_sort_key)
    index_by_qubit = {qb: idx for idx, qb in enumerate(qubits)}

    lines: list[str] = [
        "from bloqade import squin",
        "",
        "@squin.kernel(typeinfer=True, fold=True)",
        f"def {kernel_name}():",
        f"    q = squin.qalloc({len(qubits)})",
    ]

    def _operation_sort_key(op: cirq.Operation) -> tuple[int, str, tuple[int, ...]]:
        gate = op.gate
        gate_name = type(gate).__name__ if gate is not None else ""
        normalized_qubits = tuple(sorted(index_by_qubit[qb] for qb in op.qubits))
        return (len(op.qubits), gate_name, normalized_qubits)

    for moment in circuit:
        ordered_ops = sorted(moment.operations, key=_operation_sort_key)
        for op in ordered_ops:
            gate = op.gate
            if gate is None:
                raise ValueError(f"Unsupported gate-less operation: {op!r}")

            if cirq.is_measurement(op):
                raise ValueError(
                    "Measurement operations are not supported in SQUIN kernels."
                )

            if len(op.qubits) == 1:
                mat = cirq.unitary(op, default=None)
                if mat is None:
                    raise ValueError(f"Unsupported single-qubit operation: {op!r}")

                z0, y, z1 = deconstruct_single_qubit_matrix_into_angles(mat)
                # Cirq returns ZYZ angles in order (z0, y, z1) such that
                # U = Rz(z1) @ Ry(y) @ Rz(z0). SQUIN u3(theta, phi, lam) follows
                # the Rz(phi) @ Ry(theta) @ Rz(lam) convention.
                theta = y
                phi = z1
                lamb = z0
                qidx = index_by_qubit[op.qubits[0]]
                lines.append(
                    "    squin.u3("
                    f"{_format_float(float(theta))}, "
                    f"{_format_float(float(phi))}, "
                    f"{_format_float(float(lamb))}, "
                    f"q[{qidx}])"
                )
                continue

            if isinstance(gate, cirq.CZPowGate):
                if len(op.qubits) != 2:
                    raise ValueError(f"Unexpected CZ arity: {op!r}")
                if gate.exponent != 1 or gate.global_shift != 0:
                    raise ValueError(f"Only plain CZ is supported, got: {op!r}")

                q0, q1 = sorted(index_by_qubit[qb] for qb in op.qubits)
                lines.append(f"    squin.cz(q[{q0}], q[{q1}])")
                continue

            raise ValueError(
                f"Unsupported gate type in circuit: {type(gate).__name__} ({op!r})"
            )

    return "\n".join(lines)


def _qlist(qubits: list[int]) -> str:
    return "ilist.IList([" + ", ".join(f"q[{q}]" for q in qubits) + "])"


def _emit_u3_layers(ops: list[U3Op], indent: str) -> list[str]:
    # Layer by per-qubit dependency: same qubit must stay sequential.
    layers: list[list[U3Op]] = []
    used_per_layer: list[set[int]] = []
    last_layer_for_qubit: dict[int, int] = defaultdict(lambda: -1)

    for op in ops:
        start = last_layer_for_qubit[op.q] + 1
        layer_idx = start
        while layer_idx < len(layers) and op.q in used_per_layer[layer_idx]:
            layer_idx += 1
        if layer_idx == len(layers):
            layers.append([])
            used_per_layer.append(set())
        layers[layer_idx].append(op)
        used_per_layer[layer_idx].add(op.q)
        last_layer_for_qubit[op.q] = layer_idx

    out: list[str] = []
    for layer in layers:
        groups: dict[tuple[str, str, str], list[int]] = {}
        order: list[tuple[str, str, str]] = []
        for op in layer:
            key = (op.a, op.b, op.c)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(op.q)

        for key in order:
            a, b, c = key
            qs = groups[key]
            if len(qs) == 1:
                out.append(f"{indent}squin.u3({a}, {b}, {c}, q[{qs[0]}])")
            else:
                out.append(f"{indent}squin.broadcast.u3({a}, {b}, {c}, {_qlist(qs)})")
    return out


def _emit_cz_layers(ops: list[CZOp], indent: str) -> list[str]:
    # Layer by per-qubit dependency and disjointness.
    layers: list[list[CZOp]] = []
    used_per_layer: list[set[int]] = []
    last_layer_for_qubit: dict[int, int] = defaultdict(lambda: -1)

    for op in ops:
        start = max(last_layer_for_qubit[op.q0], last_layer_for_qubit[op.q1]) + 1
        layer_idx = start
        while layer_idx < len(layers):
            used = used_per_layer[layer_idx]
            if op.q0 not in used and op.q1 not in used:
                break
            layer_idx += 1

        if layer_idx == len(layers):
            layers.append([])
            used_per_layer.append(set())

        layers[layer_idx].append(op)
        used_per_layer[layer_idx].add(op.q0)
        used_per_layer[layer_idx].add(op.q1)
        last_layer_for_qubit[op.q0] = layer_idx
        last_layer_for_qubit[op.q1] = layer_idx

    out: list[str] = []
    for layer in layers:
        if len(layer) == 1:
            op = layer[0]
            out.append(f"{indent}squin.cz(q[{op.q0}], q[{op.q1}])")
        else:
            controls = [op.q0 for op in layer]
            targets = [op.q1 for op in layer]
            out.append(
                f"{indent}squin.broadcast.cz({_qlist(controls)}, {_qlist(targets)})"
            )
    return out


def parallelize_squin_blocks(source: str) -> str:
    """Parallelize contiguous u3/cz blocks in decorator-style SQUIN source."""
    lines = source.splitlines()
    output: list[str] = []
    block_kind: str | None = None
    block_ops: list[U3Op | CZOp] = []
    saw_broadcast = False

    def flush_block() -> None:
        nonlocal block_kind, block_ops, saw_broadcast
        if not block_ops:
            return
        indent = " " * 4
        if block_kind == "u3":
            emitted = _emit_u3_layers(block_ops, indent)  # type: ignore[arg-type]
        else:
            emitted = _emit_cz_layers(block_ops, indent)  # type: ignore[arg-type]
        if any(".broadcast." in line for line in emitted):
            saw_broadcast = True
        output.extend(emitted)
        block_kind = None
        block_ops = []

    for line in lines:
        m_u3 = U3_RE.match(line)
        m_cz = CZ_RE.match(line)

        if m_u3:
            kind = "u3"
            op: U3Op | CZOp = U3Op(
                m_u3.group(1),
                m_u3.group(2),
                m_u3.group(3),
                int(m_u3.group(4)),
            )
        elif m_cz:
            kind = "cz"
            op = CZOp(int(m_cz.group(1)), int(m_cz.group(2)))
        else:
            flush_block()
            output.append(line)
            continue

        if block_kind is None:
            block_kind = kind
        elif block_kind != kind:
            flush_block()
            block_kind = kind

        block_ops.append(op)

    flush_block()

    if saw_broadcast and not any(
        "from kirin.dialects import ilist" in line for line in output
    ):
        for i, line in enumerate(output):
            if line.strip() == "from bloqade import squin":
                output.insert(i + 1, "from kirin.dialects import ilist")
                break

    return "\n".join(output) + "\n"


def _simulate_cirq_statevector(circuit: cirq.Circuit) -> np.ndarray:
    qubits = sorted(circuit.all_qubits(), key=_qubit_sort_key)
    # pyqrack/bloqade treats q[0] as least-significant in basis ordering.
    # Reverse Cirq's qubit order to align statevector indexing conventions.
    qubits = list(reversed(qubits))
    sim = cirq.Simulator(dtype=np.complex128)
    result = sim.simulate(circuit, qubit_order=qubits)
    return np.asarray(result.final_state_vector, dtype=np.complex128)


def _simulate_squin_statevector(kernel_path: Path, kernel_name: str) -> np.ndarray:
    from bloqade.pyqrack.device import StackMemorySimulator

    module_name = f"_qasm_squin_verify_{kernel_path.stem}_{abs(hash(kernel_path))}"
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load generated kernel module from {kernel_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        raise ValueError(
            f"Generated kernel file {kernel_path} does not define '{kernel_name}'."
        )

    simulator = StackMemorySimulator()
    return np.asarray(simulator.state_vector(kernel), dtype=np.complex128)


def _statevectors_close(
    lhs: np.ndarray, rhs: np.ndarray, atol: float, rtol: float
) -> tuple[bool, float]:
    if lhs.shape != rhs.shape:
        return False, float("inf")

    lhs_aligned = lhs.copy()
    rhs_aligned = rhs.copy()
    overlap = np.vdot(lhs_aligned, rhs_aligned)
    if np.abs(overlap) > 1e-12:
        rhs_aligned *= np.exp(-1j * np.angle(overlap))

    diff = np.abs(lhs_aligned - rhs_aligned)
    max_abs_diff = float(np.max(diff)) if diff.size else 0.0
    return (
        bool(np.allclose(lhs_aligned, rhs_aligned, atol=atol, rtol=rtol)),
        max_abs_diff,
    )


def _write_kernel_file(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OpenQASM to a Cirq circuit, strip terminal measurements, "
            "apply Cirq parallelization, lower to decorator-style SQUIN Python, "
            "then block-parallelize SQUIN gates."
        )
    )
    parser.add_argument("qasm_file", type=Path, help="Input OpenQASM file path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Python path (default: <qasm_stem>_squin.py in current directory).",
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help="Kernel function name (default: normalized stem of qasm file).",
    )
    parser.add_argument(
        "--verify-statevector",
        action="store_true",
        help=(
            "Compare statevector for qasm->cirq (terminal measurements stripped, "
            "no parallelize) against the final generated SQUIN kernel."
        ),
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for --verify-statevector (default: 1e-6).",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for --verify-statevector (default: 1e-8).",
    )
    parser.add_argument(
        "--verify-diagnostics",
        action="store_true",
        help=(
            "Print staged comparisons using only the SQUIN simulator: "
            "unitary-cirq->squin vs parallelized-cirq->squin vs final blocked squin."
        ),
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    qasm_path = args.qasm_file.resolve()
    if not qasm_path.exists():
        parser.error(f"QASM file does not exist: {qasm_path}")

    output_path = args.output
    if output_path is None:
        output_path = Path.cwd() / f"{qasm_path.stem}_squin.py"
    output_path = output_path.resolve()

    kernel_name = args.kernel_name or qasm_path.stem
    kernel_name = _normalize_kernel_name(kernel_name)

    qasm_text = qasm_path.read_text(encoding="utf-8")
    circuit, dropped_barriers = _load_qasm_circuit(qasm_text)
    unitary_circuit, dropped_measurements = _strip_terminal_measurements(circuit)
    parallelized = parallelize(unitary_circuit)
    squin_source = circuit_to_squin_decorator_source(parallelized, kernel_name)
    transformed_source = parallelize_squin_blocks(squin_source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_kernel_file(output_path, transformed_source)
    print(f"Wrote parallelized SQUIN decorator kernel to {output_path}")
    if dropped_measurements:
        print(f"Dropped {dropped_measurements} terminal measurement operation(s).")
    if dropped_barriers:
        print(f"Dropped {dropped_barriers} barrier statement(s).")

    if args.verify_diagnostics:
        unitary_native = _to_cz_native_cirq(unitary_circuit)
        squin_from_unitary_source = circuit_to_squin_decorator_source(
            unitary_native, kernel_name
        )
        squin_from_parallelized_source = circuit_to_squin_decorator_source(
            parallelized, kernel_name
        )
        with tempfile.TemporaryDirectory(prefix="qasm_squin_diag_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            unitary_path = temp_dir_path / "unitary_squin.py"
            parallelized_path = temp_dir_path / "parallelized_squin.py"
            _write_kernel_file(unitary_path, squin_from_unitary_source)
            _write_kernel_file(parallelized_path, squin_from_parallelized_source)

            unitary_squin_state = _simulate_squin_statevector(unitary_path, kernel_name)
            parallelized_squin_state = _simulate_squin_statevector(
                parallelized_path, kernel_name
            )
            final_squin_state = _simulate_squin_statevector(output_path, kernel_name)

        up_ok, up_diff = _statevectors_close(
            unitary_squin_state,
            parallelized_squin_state,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
        pf_ok, pf_diff = _statevectors_close(
            parallelized_squin_state,
            final_squin_state,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
        uf_ok, uf_diff = _statevectors_close(
            unitary_squin_state,
            final_squin_state,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
        print("SQUIN diagnostics:")
        print(
            "  unitary-squin vs parallelized-squin: "
            f"{'PASS' if up_ok else 'FAIL'} "
            f"(max_abs_diff={up_diff:.3e})"
        )
        print(
            "  parallelized-squin vs final-blocked-squin: "
            f"{'PASS' if pf_ok else 'FAIL'} "
            f"(max_abs_diff={pf_diff:.3e})"
        )
        print(
            "  unitary-squin vs final-blocked-squin: "
            f"{'PASS' if uf_ok else 'FAIL'} "
            f"(max_abs_diff={uf_diff:.3e})"
        )

    if args.verify_statevector:
        cirq_state = _simulate_cirq_statevector(unitary_circuit)
        squin_state = _simulate_squin_statevector(output_path, kernel_name)
        ok, max_abs_diff = _statevectors_close(
            cirq_state,
            squin_state,
            atol=args.verify_atol,
            rtol=args.verify_rtol,
        )
        if not ok:
            print(
                "Statevector verification FAILED: "
                f"max_abs_diff={max_abs_diff:.3e}, "
                f"atol={args.verify_atol}, rtol={args.verify_rtol}"
            )
            return 1
        print(
            "Statevector verification passed: "
            f"max_abs_diff={max_abs_diff:.3e}, "
            f"atol={args.verify_atol}, rtol={args.verify_rtol}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
