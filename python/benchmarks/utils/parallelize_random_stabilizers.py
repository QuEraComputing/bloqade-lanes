"""Parallelize order-safe CZ runs in random_stabilizers kernels."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RANDOM_STABILIZERS_DIR = (
    Path(__file__).resolve().parent.parent / "kernels" / "random_stabilizers"
)

CZ_RE = re.compile(r"^(\s*)squin\.cz\(\s*q\[(\d+)\]\s*,\s*q\[(\d+)\]\s*\)\s*$")


@dataclass
class CZOp:
    indent: str
    ctrl: int
    tgt: int


def _parse_cz(line: str) -> CZOp | None:
    match = CZ_RE.match(line)
    if match is None:
        return None
    return CZOp(
        indent=match.group(1), ctrl=int(match.group(2)), tgt=int(match.group(3))
    )


def _emit_batch(batch: list[CZOp]) -> list[str]:
    if len(batch) == 1:
        op = batch[0]
        return [f"{op.indent}squin.cz(q[{op.ctrl}], q[{op.tgt}])"]
    controls = ", ".join(f"q[{op.ctrl}]" for op in batch)
    targets = ", ".join(f"q[{op.tgt}]" for op in batch)
    indent = batch[0].indent
    return [
        f"{indent}squin.broadcast.cz(ilist.IList([{controls}]), ilist.IList([{targets}]))"
    ]


def _parallelize_cz_block(lines: list[str]) -> tuple[list[str], bool]:
    ops: list[CZOp] = []
    for line in lines:
        op = _parse_cz(line)
        if op is None:
            return lines, False
        ops.append(op)

    out: list[str] = []
    i = 0
    used_broadcast = False
    while i < len(ops):
        batch = [ops[i]]
        used_qubits = {ops[i].ctrl, ops[i].tgt}
        i += 1
        while i < len(ops):
            candidate = ops[i]
            candidate_qubits = {candidate.ctrl, candidate.tgt}
            if used_qubits.intersection(candidate_qubits):
                break
            batch.append(candidate)
            used_qubits.update(candidate_qubits)
            i += 1
        if len(batch) > 1:
            used_broadcast = True
        out.extend(_emit_batch(batch))

    return out, used_broadcast


def parallelize_source(source: str) -> str:
    """Parallelize contiguous CZ-only blocks with adjacency-safe batching."""
    lines = source.splitlines()
    out: list[str] = []
    idx = 0
    used_broadcast = False

    while idx < len(lines):
        if _parse_cz(lines[idx]) is None:
            out.append(lines[idx])
            idx += 1
            continue

        start = idx
        while idx < len(lines) and _parse_cz(lines[idx]) is not None:
            idx += 1
        block_out, block_used_broadcast = _parallelize_cz_block(lines[start:idx])
        used_broadcast = used_broadcast or block_used_broadcast
        out.extend(block_out)

    if used_broadcast and "from kirin.dialects import ilist" not in out:
        for i, line in enumerate(out):
            if line.strip() == "from bloqade import squin":
                out.insert(i + 1, "from kirin.dialects import ilist")
                break

    return "\n".join(out) + "\n"


def parallelize_random_stabilizers(kernel_dir: Path) -> int:
    count = 0
    for path in sorted(kernel_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        original = path.read_text(encoding="utf-8")
        transformed = parallelize_source(original)
        if transformed != original:
            path.write_text(transformed, encoding="utf-8")
            count += 1
    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallelize CZ runs in random stabilizer kernels."
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=DEFAULT_RANDOM_STABILIZERS_DIR,
        help=f"Kernel directory (default: {DEFAULT_RANDOM_STABILIZERS_DIR}).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    updated = parallelize_random_stabilizers(args.kernel_dir.resolve())
    print(f"Updated {updated} random stabilizer kernel file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
