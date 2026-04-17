"""Generate random stabilizer SQuin kernels from gemini_benchmarking circuits."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

import stim

DEFAULT_GEMINI_REPO = Path("/Users/jasonludmir/Documents/gemini_benchmarking")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "kernels" / "random_stabilizers"
)


def _load_multi_qubit_rb(gemini_repo: Path):
    sys.path.insert(0, str(gemini_repo))
    sys.path.insert(0, str(gemini_repo / "src"))
    stabilizer_module = importlib.import_module("programs.stabilizer_benchmarking")
    return stabilizer_module.multi_qubit_rb


def _active_qubit_map(circuit: stim.Circuit) -> dict[int, int]:
    active: set[int] = set()
    for op in circuit:
        target_groups = getattr(op, "target_groups", None)
        if target_groups is None:
            continue
        for group in target_groups():
            for target in group:
                active.add(target.value)
    return {q: i for i, q in enumerate(sorted(active))}


def render_squin_kernel_module(*, kernel_name: str, circuit: stim.Circuit) -> str:
    """Render one plain decorator-style SQuin kernel module."""
    qubit_map = _active_qubit_map(circuit)
    lines: list[str] = [
        '"""Generated from gemini_benchmarking multi_qubit_rb circuits."""',
        "",
        "from __future__ import annotations",
        "",
        "# pyright: reportCallIssue=false",
        "",
        "from bloqade import squin",
        "",
        "@squin.kernel(typeinfer=True, fold=True)",
        f"def {kernel_name}():",
        f"    q = squin.qalloc({len(qubit_map)})",
    ]
    for op in circuit:
        if op.name == "TICK":
            continue
        target_groups = getattr(op, "target_groups", None)
        if target_groups is None:
            continue
        for group in target_groups():
            q0 = qubit_map[group[0].value]
            if op.name == "X":
                lines.append(f"    squin.x(q[{q0}])")
            elif op.name == "Y":
                lines.append(f"    squin.y(q[{q0}])")
            elif op.name == "Z":
                lines.append(f"    squin.z(q[{q0}])")
            elif op.name == "S":
                lines.append(f"    squin.s(q[{q0}])")
            elif op.name == "S_DAG":
                lines.append(f"    squin.s(q[{q0}], adjoint=True)")
            elif op.name == "SQRT_X":
                lines.append(f"    squin.sqrt_x(q[{q0}])")
            elif op.name == "SQRT_X_DAG":
                lines.append(f"    squin.sqrt_x_adj(q[{q0}])")
            elif op.name == "SQRT_Y":
                lines.append(f"    squin.sqrt_y(q[{q0}])")
            elif op.name == "SQRT_Y_DAG":
                lines.append(f"    squin.sqrt_y_adj(q[{q0}])")
            elif op.name == "CZ":
                q1 = qubit_map[group[1].value]
                lines.append(f"    squin.cz(q[{q0}], q[{q1}])")
            else:
                raise ValueError(f"Unsupported stim instruction: {op.name!r}")
    lines.append("")
    return "\n".join(lines)


def generate_random_stabilizers(*, gemini_repo: Path, output_dir: Path) -> int:
    """Generate all random stabilizer kernel files into the output directory."""
    multi_qubit_rb = _load_multi_qubit_rb(gemini_repo)
    circuits, _ = multi_qubit_rb()
    kernel_names = sorted(circuits)

    output_dir.mkdir(parents=True, exist_ok=True)

    provenance_dir = output_dir / "_provenance"
    if provenance_dir.exists():
        shutil.rmtree(provenance_dir)

    desired_files = {"__init__.py"} | {f"{name}.py" for name in kernel_names}
    for existing in output_dir.glob("*.py"):
        if existing.name not in desired_files:
            existing.unlink()

    for name in kernel_names:
        source = render_squin_kernel_module(kernel_name=name, circuit=circuits[name])
        (output_dir / f"{name}.py").write_text(source, encoding="utf-8")

    return len(kernel_names)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate random stabilizer kernels from gemini_benchmarking."
    )
    parser.add_argument(
        "--gemini-repo",
        type=Path,
        default=DEFAULT_GEMINI_REPO,
        help=f"Path to gemini_benchmarking checkout (default: {DEFAULT_GEMINI_REPO}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated kernels (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    count = generate_random_stabilizers(
        gemini_repo=args.gemini_repo.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(
        f"Generated {count} random stabilizer kernels in {args.output_dir.resolve()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
