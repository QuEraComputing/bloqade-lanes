"""Side-by-side comparison of DefaultTargetGenerator vs
CongestionAwareTargetGenerator on the logical Gemini arch.

All scenarios start with atoms packed into the LEFT (even-numbered)
words of the spec, leaving the right (odd-numbered) words — the CZ
blockade partners — free. For each CZ stage, both generators are
asked for a target placement; this script prints the two side-by-side
and annotates which qubit moved, in which direction (L/R), and how
far along the word axis.

Run with:  uv run python demo/congestion_aware_compare.py
"""

from __future__ import annotations

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.heuristics.physical.target_generator import (
    CongestionAwareTargetGenerator,
    DefaultTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
)


def left_word_placement(
    arch: layout.ArchSpec, n_qubits: int
) -> tuple[layout.LocationAddress, ...]:
    """Pack n_qubits into the first n_qubits even-indexed home words."""
    left_homes = sorted(
        (s for s in arch.home_sites if s.word_id % 2 == 0),
        key=lambda s: s.word_id,
    )
    if n_qubits > len(left_homes):
        raise ValueError(
            f"arch has only {len(left_homes)} left-word homes, requested {n_qubits}"
        )
    return tuple(left_homes[:n_qubits])


def build_ctx(
    arch: layout.ArchSpec,
    n_qubits: int,
    controls: tuple[int, ...],
    targets: tuple[int, ...],
) -> TargetContext:
    layout_tup = left_word_placement(arch, n_qubits)
    return TargetContext(
        arch_spec=arch,
        state=ConcreteState(
            occupied=frozenset(),
            layout=layout_tup,
            move_count=(0,) * len(layout_tup),
        ),
        controls=controls,
        targets=targets,
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )


def _move_description(
    qid: int,
    before: layout.LocationAddress,
    after: layout.LocationAddress,
) -> str:
    if before == after:
        return f"q{qid} stays @ w{before.word_id}"
    dw = after.word_id - before.word_id
    direction = "→R" if dw > 0 else "←L"
    return f"q{qid}: w{before.word_id} {direction} w{after.word_id}  (Δw={dw:+d})"


def _summarize_candidate(
    label: str,
    ctx: TargetContext,
    gen: TargetGeneratorABC,
) -> list[str]:
    candidates = gen.generate(ctx)
    if not candidates:
        return [f"{label}: [] (defers to default fallback)"]
    if len(candidates) > 1:
        lines = [f"{label}: {len(candidates)} candidates"]
    else:
        lines = [f"{label}:"]
    placement = ctx.placement
    for i, cand in enumerate(candidates):
        if len(candidates) > 1:
            lines.append(f"  candidate {i}:")
        for ctrl, tgt in zip(ctx.controls, ctx.targets):
            mover_qid, mover_before, mover_after = None, None, None
            for qid in (ctrl, tgt):
                if cand[qid] != placement[qid]:
                    mover_qid = qid
                    mover_before = placement[qid]
                    mover_after = cand[qid]
                    break
            if mover_qid is None:
                lines.append(
                    f"    pair (c=q{ctrl}, t=q{tgt}): no move (already partnered)"
                )
            else:
                role = "ctrl" if mover_qid == ctrl else "tgt"
                desc = _move_description(mover_qid, mover_before, mover_after)
                lines.append(f"    pair (c=q{ctrl}, t=q{tgt}) [{role} moves]: {desc}")
    return lines


def compare(title: str, ctx: TargetContext) -> None:
    print(f"\n=== {title} ===")
    print(
        f"initial: {[f'q{i}@w{loc.word_id}' for i, loc in enumerate(ctx.state.layout)]}"
    )
    print(f"CZ pairs: {list(zip(ctx.controls, ctx.targets))}")
    default_lines = _summarize_candidate("Default", ctx, DefaultTargetGenerator())
    cong_lines = _summarize_candidate(
        "Congestion-aware", ctx, CongestionAwareTargetGenerator()
    )
    for line in default_lines:
        print(line)
    print()
    for line in cong_lines:
        print(line)


def main() -> None:
    arch = logical.get_arch_spec()

    # Scenario 1: single pair, adjacent left words (q0 @ w0, q1 @ w2).
    # Default moves q0 (control) to partner(w2) = w3.
    compare(
        "Scenario 1: single pair, adjacent left words",
        build_ctx(arch, n_qubits=2, controls=(0,), targets=(1,)),
    )

    # Scenario 2: single pair, far-apart left words (q0 @ w0, q1 @ w18).
    # Default moves q0 to partner(w18) = w19 (big rightward jump).
    # Congestion-aware has a symmetric alternative (move q1 to partner(w0) = w1),
    # but both directions will tie on cost so tiebreak picks control.
    compare(
        "Scenario 2: single pair, far-apart left words",
        build_ctx(arch, n_qubits=10, controls=(0,), targets=(9,)),
    )

    # Scenario 3: two independent pairs, natural ordering.
    # q0@w0 ↔ q1@w2 and q2@w4 ↔ q3@w6. Both generators should pick the
    # control direction for each pair (tiebreak).
    compare(
        "Scenario 3: two independent pairs, natural order",
        build_ctx(arch, n_qubits=4, controls=(0, 2), targets=(1, 3)),
    )

    # Scenario 4: two pairs that reach across each other.
    # q0@w0, q1@w2, q2@w4, q3@w6 with pairs (q0,q3) and (q1,q2).
    # The "inner" pair (q1,q2) sits inside the "outer" pair (q0,q3).
    # Paths for the two pairs may share lanes; congestion-aware should
    # at least match default on the logical arch (no lane-direction
    # conflicts arise) but this exercises the multi-pair commit loop.
    compare(
        "Scenario 4: nested pairs (outer q0-q3, inner q1-q2)",
        build_ctx(arch, n_qubits=4, controls=(0, 1), targets=(3, 2)),
    )


if __name__ == "__main__":
    main()
