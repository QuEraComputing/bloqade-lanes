"""Inline SQuIn kernel catalog for movement/layering/CZ regression coverage.

This module is intentionally kernels-only: it does not contain pytest tests.
Future parametrized tests can consume this catalog as follows:

    from tests._squin_kernels import select_kernels

    for spec in select_kernels(tags={"movement", "cz"}):
        kernel = spec.build_kernel()
        # compile and assert invariants in a test module

Rubric:
- Qubit hard limit: <= 10 for every kernel.
- Mixed composition:
  - curated motifs to stress known compiler behaviors
  - manually translated qasmbench-inspired kernels
- Required tags:
  movement, layering, cz, control, entangling_density, locality_pattern
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable, Literal

from kirin.dialects import ilist

from bloqade import squin

Tag = Literal[
    "movement",
    "layering",
    "cz",
    "control",
    "entangling_density",
    "locality_pattern",
]

ALLOWED_TAGS: frozenset[str] = frozenset(
    {
        "movement",
        "layering",
        "cz",
        "control",
        "entangling_density",
        "locality_pattern",
    }
)


@dataclass(frozen=True)
class KernelSpec:
    name: str
    origin: Literal["curated", "qasmbench_manual"]
    source_id: str
    qubits: int
    tags: tuple[Tag, ...]
    build_kernel: Callable[[], object]


def _curated_chain_cz_sweep() -> object:
    """Chain-like CZ pairings in two passes with global rotations around them.

    Intended stress: ensure CZ batches compile cleanly as adjacency patterns shift
    from (0-1, 2-3, 4-5) to (1-2, 3-4), without corrupting layer structure.
    """

    @squin.kernel
    def kernel():
        reg = squin.qalloc(6)
        squin.broadcast.u3(0.2 * pi, 0.0, 0.1 * pi, reg)
        squin.broadcast.cz(
            ilist.IList([reg[0], reg[2], reg[4]]),
            ilist.IList([reg[1], reg[3], reg[5]]),
        )
        squin.broadcast.cz(
            ilist.IList([reg[1], reg[3]]),
            ilist.IList([reg[2], reg[4]]),
        )
        squin.broadcast.u3(0.15 * pi, 0.25 * pi, 0.0, reg)

    return kernel


def _curated_star_fanout() -> object:
    """Star/fanout control pattern with sequential CZ interactions from a shared hub.

    Intended stress: repeated reuse of one control qubit across many interactions,
    exercising non-parallel entangler sequencing.
    """

    @squin.kernel
    def kernel():
        reg = squin.qalloc(5)
        squin.h(reg[0])
        for i in range(1, len(reg)):
            squin.cx(reg[0], reg[i])
        squin.cz(reg[0], reg[1])
        squin.cz(reg[0], reg[3])

    return kernel


def _curated_movement_reorder_pattern() -> object:
    """Broadcast CZ blocks with intermediate targeted CNOT reordering pressure.

    Intended stress: movement-sensitive scheduling where local interaction edits
    occur between larger CZ batches, probing reorder/regression behavior.
    """

    @squin.kernel
    def kernel():
        reg = squin.qalloc(7)
        squin.broadcast.u3(0.5 * pi, 0.0, 0.0, reg)
        squin.broadcast.cz(
            ilist.IList([reg[0], reg[1], reg[2]]),
            ilist.IList([reg[3], reg[4], reg[5]]),
        )
        squin.cx(reg[6], reg[2])
        squin.cx(reg[6], reg[4])
        squin.broadcast.cz(
            ilist.IList([reg[0], reg[2], reg[4]]),
            ilist.IList([reg[1], reg[3], reg[5]]),
        )

    return kernel


def _curated_control_entangler_mix() -> object:
    """Mixed sparse controls and overlapping entangler neighborhoods.

    Intended stress: interplay between CNOT-style control paths and CZ-style
    entangler layers with partial qubit overlap across consecutive operations.
    """

    @squin.kernel
    def kernel():
        reg = squin.qalloc(8)
        for idx in (0, 2, 5):
            squin.h(reg[idx])
        squin.cx(reg[0], reg[3])
        squin.cx(reg[2], reg[4])
        squin.cx(reg[5], reg[7])
        squin.broadcast.cz(ilist.IList([reg[3], reg[6]]), ilist.IList([reg[4], reg[7]]))
        squin.cz(reg[4], reg[6])
        squin.broadcast.u3(0.2 * pi, 0.5 * pi, 0.0, reg)

    return kernel


def _qasmbench_toffoli_n3_manual() -> object:
    """Manual translation inspired by QASMBench's toffoli_n3 circuit"""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.h(reg[2])
        squin.cx(reg[1], reg[2])
        squin.broadcast.u3(0.0, 0.75 * pi, 0.25 * pi, ilist.IList([reg[2]]))
        squin.cx(reg[0], reg[2])
        squin.cx(reg[1], reg[2])
        squin.cx(reg[0], reg[2])
        squin.cx(reg[0], reg[1])
        squin.h(reg[2])

    return kernel


def _qasmbench_fredkin_n3_manual() -> object:
    """Manual translation inspired by QASMBench's fredkin_n3 circuit"""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.cx(reg[2], reg[1])
        squin.cx(reg[0], reg[1])
        squin.h(reg[2])
        squin.cx(reg[0], reg[2])
        squin.cx(reg[2], reg[1])
        squin.cx(reg[0], reg[2])
        squin.cx(reg[2], reg[1])
        squin.h(reg[2])

    return kernel


def _qasmbench_qft_n4_manual() -> object:
    """Manual translation inspired by QASMBench's qft_n4 circuit"""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(4)
        squin.h(reg[0])
        squin.cz(reg[1], reg[0])
        squin.cz(reg[2], reg[0])
        squin.cz(reg[3], reg[0])
        squin.h(reg[1])
        squin.cz(reg[2], reg[1])
        squin.cz(reg[3], reg[1])
        squin.h(reg[2])
        squin.cz(reg[3], reg[2])
        squin.h(reg[3])

    return kernel


def _qasmbench_qaoa_n3_manual() -> object:
    """Manual translation inspired by QASMBench's qaoa_n3 circuit"""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.broadcast.u3(0.5 * pi, 0.0, pi, reg)
        squin.cx(reg[0], reg[2])
        squin.cx(reg[0], reg[1])
        squin.cx(reg[1], reg[2])
        squin.broadcast.u3(0.18 * pi, 0.0, 0.0, ilist.IList([reg[2]]))
        squin.cx(reg[1], reg[2])
        squin.cx(reg[0], reg[1])
        squin.broadcast.u3(0.11 * pi, 0.0, 0.0, reg)

    return kernel


def _qasmbench_basis_change_n3_manual() -> object:
    """Manual translation inspired by QASMBench's basis_change_n3 circuit"""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.broadcast.u3(0.5 * pi, 0.0, 0.06 * pi, ilist.IList([reg[2]]))
        squin.broadcast.u3(0.5 * pi, 1.5 * pi, 0.29 * pi, ilist.IList([reg[1]]))
        squin.broadcast.u3(0.5 * pi, 1.5 * pi, 1.5 * pi, ilist.IList([reg[0]]))
        squin.cz(reg[1], reg[2])
        squin.cz(reg[0], reg[1])
        squin.broadcast.u3(0.12 * pi, 0.5 * pi, 1.5 * pi, ilist.IList([reg[1], reg[2]]))
        squin.broadcast.cz(ilist.IList([reg[0]]), ilist.IList([reg[1]]))
        squin.broadcast.u3(0.33 * pi, 0.0, 0.0, ilist.IList([reg[0], reg[1]]))

    return kernel


def _qasmbench_adder_n4_manual() -> object:
    """Manual translation inspired by QASMBench's adder_n4 circuit"""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(4)
        squin.h(reg[3])
        squin.cx(reg[2], reg[3])
        squin.cx(reg[0], reg[1])
        squin.cx(reg[2], reg[3])
        squin.cx(reg[3], reg[0])
        squin.cx(reg[1], reg[2])
        squin.cx(reg[0], reg[1])
        squin.cx(reg[2], reg[3])
        squin.cx(reg[0], reg[1])
        squin.cx(reg[2], reg[3])
        squin.cx(reg[3], reg[0])
        squin.h(reg[3])

    return kernel


def _qasmbench_qpe_n9_manual() -> object:
    """Manual translation inspired by QASMBench's qpe_n9 circuit"""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(9)
        squin.broadcast.u3(
            0.5 * pi,
            0.0,
            pi,
            ilist.IList([reg[0], reg[1], reg[2], reg[3], reg[4], reg[5]]),
        )
        for i in (6, 7, 8):
            squin.x(reg[i])
        squin.cx(reg[5], reg[7])
        squin.cz(reg[7], reg[8])
        squin.cx(reg[5], reg[7])
        squin.cz(reg[5], reg[0])
        squin.cz(reg[4], reg[0])
        squin.cz(reg[3], reg[1])
        squin.cz(reg[2], reg[1])
        squin.cz(reg[1], reg[0])
        for i in range(6):
            squin.h(reg[i])

    return kernel


SQUIN_KERNEL_SUITE: tuple[KernelSpec, ...] = (
    KernelSpec(
        name="curated_chain_cz_sweep",
        origin="curated",
        source_id="curated.chain_cz_sweep",
        qubits=6,
        tags=("movement", "layering", "cz", "entangling_density", "locality_pattern"),
        build_kernel=_curated_chain_cz_sweep,
    ),
    KernelSpec(
        name="curated_star_fanout",
        origin="curated",
        source_id="curated.star_fanout",
        qubits=5,
        tags=("movement", "control", "layering", "locality_pattern"),
        build_kernel=_curated_star_fanout,
    ),
    KernelSpec(
        name="curated_movement_reorder_pattern",
        origin="curated",
        source_id="curated.movement_reorder_pattern",
        qubits=7,
        tags=("movement", "layering", "cz", "control", "locality_pattern"),
        build_kernel=_curated_movement_reorder_pattern,
    ),
    KernelSpec(
        name="curated_control_entangler_mix",
        origin="curated",
        source_id="curated.control_entangler_mix",
        qubits=8,
        tags=("movement", "layering", "cz", "control", "entangling_density"),
        build_kernel=_curated_control_entangler_mix,
    ),
    KernelSpec(
        name="qasmbench_toffoli_n3_manual",
        origin="qasmbench_manual",
        source_id="toffoli_n3",
        qubits=3,
        tags=("control", "layering", "movement", "locality_pattern"),
        build_kernel=_qasmbench_toffoli_n3_manual,
    ),
    KernelSpec(
        name="qasmbench_fredkin_n3_manual",
        origin="qasmbench_manual",
        source_id="fredkin_n3",
        qubits=3,
        tags=("control", "layering", "movement", "locality_pattern"),
        build_kernel=_qasmbench_fredkin_n3_manual,
    ),
    KernelSpec(
        name="qasmbench_qft_n4_manual",
        origin="qasmbench_manual",
        source_id="qft_n4",
        qubits=4,
        tags=("layering", "cz", "entangling_density", "locality_pattern"),
        build_kernel=_qasmbench_qft_n4_manual,
    ),
    KernelSpec(
        name="qasmbench_qaoa_n3_manual",
        origin="qasmbench_manual",
        source_id="qaoa_n3",
        qubits=3,
        tags=("layering", "cz", "entangling_density", "movement"),
        build_kernel=_qasmbench_qaoa_n3_manual,
    ),
    KernelSpec(
        name="qasmbench_basis_change_n3_manual",
        origin="qasmbench_manual",
        source_id="basis_change_n3",
        qubits=3,
        tags=("cz", "layering", "movement", "locality_pattern"),
        build_kernel=_qasmbench_basis_change_n3_manual,
    ),
    KernelSpec(
        name="qasmbench_adder_n4_manual",
        origin="qasmbench_manual",
        source_id="adder_n4",
        qubits=4,
        tags=("control", "movement", "layering", "locality_pattern"),
        build_kernel=_qasmbench_adder_n4_manual,
    ),
    KernelSpec(
        name="qasmbench_qpe_n9_manual",
        origin="qasmbench_manual",
        source_id="qpe_n9",
        qubits=9,
        tags=("control", "cz", "layering", "movement", "entangling_density"),
        build_kernel=_qasmbench_qpe_n9_manual,
    ),
)


def select_kernels(
    tags: set[str] | None = None,
    max_qubits: int = 10,
    origins: set[str] | None = None,
) -> tuple[KernelSpec, ...]:
    """Filter kernels by tag/origin while honoring the qubit cap."""
    selected: list[KernelSpec] = []
    for spec in SQUIN_KERNEL_SUITE:
        if spec.qubits > max_qubits:
            continue
        if tags is not None and not tags.issubset(set(spec.tags)):
            continue
        if origins is not None and spec.origin not in origins:
            continue
        selected.append(spec)
    return tuple(selected)
