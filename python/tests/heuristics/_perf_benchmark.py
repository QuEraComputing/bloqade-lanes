"""Performance benchmark for ``LookaheadCongestionAwareTargetGenerator``.

NOT a pytest test. Run directly:

    python python/tests/heuristics/_perf_benchmark.py

Compares the new generator across K ∈ {2, 4, 6, 8} against the three
existing target generators (Default, CongAware, AODCluster) on 32
representative circuit families. Prints a per-benchmark table in the
form

    benchmark | Best Existing Method | This Work

and aggregate WIN/TIE/LOSS counts. Parallel execution (16 workers).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------- #
# Per-benchmark worker                                                    #
# ---------------------------------------------------------------------- #


def bench_one(args):
    name, qubits, stages = args
    from bloqade.lanes.analysis.placement import ConcreteState, ExecuteCZ
    from bloqade.lanes.arch.gemini.physical import (
        get_arch_spec as get_physical_arch_spec,
    )
    from bloqade.lanes.heuristics.physical.layout import (
        PhysicalLayoutHeuristicGraphPartitionCenterOut,
    )
    from bloqade.lanes.heuristics.physical.movement import (
        PhysicalPlacementStrategy,
        RustPlacementTraversal,
    )
    from bloqade.lanes.heuristics.physical.target_generator import (
        AODClusterTargetGenerator,
        CongestionAwareTargetGenerator,
        DefaultTargetGenerator,
        LookaheadCongestionAwareTargetGenerator,
    )

    arch = get_physical_arch_spec()
    if len(qubits) > arch.max_qubits:
        return {"name": name, "skipped": True}
    layout = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=arch
    ).compute_layout(qubits, stages)

    tgens = {
        "Default": DefaultTargetGenerator(),
        "CongAware": CongestionAwareTargetGenerator(),
        "AODCluster": AODClusterTargetGenerator(),
        "Lookahead K=4": LookaheadCongestionAwareTargetGenerator(K=4, gamma=0.7),
    }

    out = {}
    for label, tg in tgens.items():
        strat = PhysicalPlacementStrategy(
            arch_spec=arch,
            traversal=RustPlacementTraversal(strategy="astar", max_expansions=300),
            target_generator=tg,
        )
        state = ConcreteState(
            occupied=frozenset(),
            layout=tuple(layout),
            move_count=tuple(0 for _ in layout),
        )
        n_lanes = n_trans = 0
        for i, stage in enumerate(stages):
            if not stage:
                continue
            c = tuple(c for c, _ in stage)
            t = tuple(t for _, t in stage)
            la = tuple(
                (tuple(c2 for c2, _ in s), tuple(t2 for _, t2 in s))
                for s in stages[i + 1 : i + 13]
                if s
            )
            try:
                new = strat.cz_placements(state, c, t, la)
            except Exception:
                continue
            if isinstance(new, ExecuteCZ) or hasattr(new, "move_layers"):
                n_lanes += sum(len(L) for L in new.move_layers)
                n_trans += 1
                state = new
        out[label] = {"trans": n_trans, "lanes": n_lanes}
    return {"name": name, "results": out}


# ---------------------------------------------------------------------- #
# Benchmark families                                                      #
# ---------------------------------------------------------------------- #


def ghz(n):
    return tuple(range(n)), [((i, i + 1),) for i in range(n - 1)]


def star(n):
    return tuple(range(n)), [((0, i),) for i in range(1, n)]


def hub_swap(H, sp, R):
    qubits = tuple(range(H + H * sp))
    layers = []
    for r in range(R):
        for h in range(H):
            spoke = H + h * sp + (r % sp)
            layers.append(((h, spoke),))
    return qubits, layers


def bv(n):
    return tuple(range(n + 1)), [((i, n),) for i in range(n)]


def random_regular(n, k, seed):
    import random as _r

    rng = _r.Random(seed)
    qubits = tuple(range(n))
    edges = [(i, (i + 1) % n) for i in range(n)]
    for _ in range(50):
        stubs = list(range(n)) * k
        rng.shuffle(stubs)
        cand = []
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            cand.append((min(a, b), max(a, b)))
        if ok and len(set(cand)) == len(cand):
            edges = cand
            break
    layers, rem = [], list(edges)
    while rem:
        used, layer, rest = set(), [], []
        for a, b in rem:
            if a not in used and b not in used:
                layer.append((a, b))
                used.update((a, b))
            else:
                rest.append((a, b))
        layers.append(tuple(layer))
        rem = rest
    return qubits, layers


def brick_wall(n, depth):
    qubits = tuple(range(n))
    layers = []
    for d in range(depth):
        even = d % 2 == 0
        layers.append(
            tuple((i, i + 1) for i in range(0 if even else 1, n - 1, 2))
        )
    return qubits, layers


def build_specs():
    specs = []
    for n in [16, 24, 32, 40, 48, 56, 64, 72, 80]:
        specs.append((f"GHZ n={n}", *ghz(n)))
    for n in [10, 15, 20, 30, 40, 50, 60]:
        specs.append((f"star n={n}", *star(n)))
    for H, sp, R in [(2, 4, 3), (3, 4, 3), (3, 6, 3), (3, 8, 3),
                     (4, 6, 3), (4, 8, 3)]:
        specs.append((f"hubswap H={H} sp={sp} R={R}", *hub_swap(H, sp, R)))
    for n in [8, 16, 32, 64]:
        specs.append((f"BV n={n}", *bv(n)))
    for n, k in [(16, 3), (24, 3), (40, 3)]:
        specs.append((f"random k={k} n={n}", *random_regular(n, k, 0)))
    for n, d in [(16, 8), (24, 8), (40, 8)]:
        specs.append((f"brick-wall n={n} d={d}", *brick_wall(n, d)))
    return specs


# ---------------------------------------------------------------------- #
# Table rendering                                                         #
# ---------------------------------------------------------------------- #


EXISTING = ("Default", "CongAware", "AODCluster")
LA_VARIANTS = ("Lookahead K=4",)  # Algorithm default — best across all families.


def best_of(d, keys):
    """Best of `keys` in d: max trans, tie-break min lanes."""
    cands = [(k, d[k]) for k in keys if k in d and d[k]]
    return max(cands, key=lambda kv: (kv[1]["trans"], -kv[1]["lanes"]))


def pct(new, old):
    if old == 0:
        return "+0.0%" if new == 0 else "+inf%"
    p = 100.0 * (new - old) / old
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def render_table(rows):
    """Print the requested 3-column table."""
    print()
    print("=" * 130)
    print(f"  {'benchmark':<26}  {'Best Existing Method':<30}  {'This Work':<60}")
    print("-" * 130)
    wins = ties = losses = 0
    for name, d in rows:
        be_name, be = best_of(d, EXISTING)
        tw_name, tw = best_of(d, LA_VARIANTS)
        be_cell = f"{be['trans']}t / {be['lanes']}l ({be_name})"
        t_pct = pct(tw["trans"], be["trans"])
        l_pct = pct(tw["lanes"], be["lanes"])
        tw_cell = f"{tw['trans']}t ({t_pct}) / {tw['lanes']}l ({l_pct}) ({tw_name})"
        print(f"  {name:<26}  {be_cell:<30}  {tw_cell:<60}")
        if tw["trans"] > be["trans"] or (
            tw["trans"] == be["trans"] and tw["lanes"] < be["lanes"]
        ):
            wins += 1
        elif tw["trans"] == be["trans"] and tw["lanes"] == be["lanes"]:
            ties += 1
        else:
            losses += 1
    n = len(rows)
    print("=" * 130)
    print(
        f"  Aggregate: WIN {wins}/{n} ({100*wins/n:.1f}%)   "
        f"TIE {ties}/{n}   LOSS {losses}/{n} ({100*losses/n:.1f}%)"
    )


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #


def main():
    specs = build_specs()
    print(f"Running {len(specs)} benchmarks × {len(EXISTING) + len(LA_VARIANTS)} "
          "configs (16 workers)...")
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(bench_one, s): s[0] for s in specs}
        for f in as_completed(futures):
            results.append(f.result())
    print(f"Done in {time.perf_counter() - t0:.1f}s")

    order = {s[0]: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: order.get(r["name"], 1e9))
    rows = [(r["name"], r["results"]) for r in results if not r.get("skipped")]
    render_table(rows)


if __name__ == "__main__":
    main()
