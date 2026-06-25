# [[8,3,2]] Logical-Gadget Demo Design

**Goal:** Rewrite `demo/move_demo.py` into a worked example that uses the
movement-level primitives (`movement.permute`, `movement.move_to`,
`movement.cz_partner`, `qubit.new_at`) to implement **physically correct**
logical gadgets for the [[8,3,2]] code: a slot-based allocator that prepares the
logical-zero state, transversal Clifford gadgets, the signature transversal
non-Clifford gate, and an intra-block logical relabel driven by `movement.permute`
with a top-level `VIRTUAL` toggle that enables/disables the physical move
instructions.

**Non-goals:** No state simulation / fidelity verification (compile + visualize
only). No new dialect statements or compiler passes — this is a demo built on the
existing public API.

**Tech stack:** `bloqade.squin` (gates: `h`, `cz`, `cx`, `t`, `t_adj`, and their
`broadcast` forms), `bloqade.gemini.physical` kernels, `bloqade.gemini.common.dialects`
(`movement`, `qubit`), `bloqade.lanes.pipeline.PhysicalPipeline`,
`bloqade.lanes.heuristics.physical.make_physical_placement_strategy`,
`bloqade.lanes.visualize.debugger`, `kirin.dialects.ilist`.

---

## 1. [[8,3,2]] Conventions (expert-checkable)

These conventions are fixed up front and documented in the demo's module
docstring so the physics is auditable.

- **Qubits on cube vertices.** Vertex `v = (x, y, z)`, `x,y,z ∈ {0,1}`, integer
  label `n = 4x + 2y + z`, giving qubits `0..7`:

  | n | (x,y,z) | parity x⊕y⊕z |
  |---|---------|--------------|
  | 0 | (0,0,0) | 0 (even) |
  | 1 | (0,0,1) | 1 (odd)  |
  | 2 | (0,1,0) | 1 (odd)  |
  | 3 | (0,1,1) | 0 (even) |
  | 4 | (1,0,0) | 1 (odd)  |
  | 5 | (1,0,1) | 0 (even) |
  | 6 | (1,1,0) | 0 (even) |
  | 7 | (1,1,1) | 1 (odd)  |

  **Even-parity set `{0,3,5,6}`, odd-parity set `{1,2,4,7}`** (the cube's
  bipartite 2-coloring).

- **Stabilizers** (`n − k = 5` generators):
  - One weight-8 X (cell): `S_X = X^{⊗8}`.
  - Four independent weight-4 Z (faces). The six faces are
    `x=0:{0,1,2,3}`, `x=1:{4,5,6,7}`, `y=0:{0,1,4,5}`, `y=1:{2,3,6,7}`,
    `z=0:{0,2,4,6}`, `z=1:{1,3,5,7}`; their GF(2) rank is 4 (relations
    `(x0)+(x1) = (y0)+(y1) = (z0)+(z1) = 1^8`).

- **Logical operators** (axis `i ∈ {1,2,3}` ↔ `x,y,z`):
  - `X̄_i` = X on the face perpendicular to axis `i` (weight 4):
    `X̄_1 = X{0,1,2,3}`, `X̄_2 = X{0,1,4,5}`, `X̄_3 = X{0,2,4,6}`.
  - `Z̄_i` = Z on an edge along axis `i` (weight 2):
    `Z̄_1 = Z{0,4}`, `Z̄_2 = Z{0,2}`, `Z̄_3 = Z{0,1}`.
  - `X̄_i` anticommutes only with `Z̄_i` (single overlap at qubit 0), commutes
    with the others.

- **Logical zero.** With all stabilizers `+1` and all `Z̄_i = +1`, the only
  computational basis states are `00000000` and `11111111`; symmetrized by
  `S_X = X^{⊗8}`, therefore
  **`|000⟩_L = (|0⟩^{⊗8} + |1⟩^{⊗8}) / √2` = the 8-qubit GHZ state.**

---

## 2. Architecture

Follows the `demo/steane_demo.py` allocator-factory pattern:

```
eight_three_two_allocator()  ->  (qalloc, qalloc_slot)   # factory closure over slot layout
flat(block)                                                # IList[LogicalQubit] -> IList[Qubit]
init_logical_zero(reg)                                     # GHZ prep, |000>_L
transversal_cx(controls, targets)                          # logical CX between two blocks
logical_ccz(reg)                                           # T-even / T†-odd -> CCZ_123
logical_swap(reg)                                          # SWAP logical 1<->2 via permute
measure_logical_block(blocks)                              # flatten + single terminal measure
main()                                                     # compose + measure
```

A `LogicalBlock` type alias = `ilist.IList[Qubit, Literal[8]]`. The factory holds
the canonical slot list (each slot = 8 `(zone, word, site)` addresses) so slot
layout is defined once and gadgets are layout-agnostic.

**Module-level toggle:** `VIRTUAL: bool` — read by `logical_swap`. `False` →
emit physical `movement.permute`; `True` → relabel only (no physical move).

---

## 3. File Layout

| File | Status | Responsibility |
|------|--------|----------------|
| `demo/move_demo.py` | **REWRITE** | Entire [[8,3,2]] gadget demo (replaces the current rotate-and-entangle scratch demo) |

No other files change.

---

## 4. Components

### 4.1 Allocator + `|000⟩_L` init

`eight_three_two_allocator()` returns `(qalloc, qalloc_slot)`:

- **Canonical slots.** A list of slots; each slot is 8 addresses
  `(zone, word, site)`. Layout convention: a slot occupies one zone-0 word-block
  with the 8 vertices laid across `(word, site)` so the parity 2-coloring lands on
  alternating, move-friendly positions. Concrete word/site assignment follows the
  existing demo's `(0, word, site)` scheme; the exact words per slot are an
  implementation detail (chosen for placement parallelism), not a correctness
  concern.
- **`qalloc_slot(slot_index)`** — `ilist.map(qubit.new_at, addresses)` to allocate
  the 8 physical qubits, then `init_logical_zero(reg)`; returns the 8-qubit block.
- **`qalloc(slot_indices)`** — `ilist.map` `qalloc_slot` over slot indices, returns
  an `IList[LogicalBlock]`.

**`init_logical_zero(reg)`** prepares GHZ$_8$ = `|000⟩_L`:
- `h` on a root qubit, then `cz`/`cx` along a star or chain graph so the result is
  `(|0⟩^{⊗8} + |1⟩^{⊗8})/√2`. The exact entangling schedule is chosen to be
  move-friendly (nearest-neighbour staging via `move_to`/`cz_partner`); the
  asserted invariant is **prepared state = GHZ = `|000⟩_L`**, which is
  schedule-independent.

### 4.2 `transversal_cx(controls, targets)`

`squin.broadcast.cx(flat(controls), flat(targets))` — physical CX on the 8
aligned qubit pairs between two [[8,3,2]] blocks. Because [[8,3,2]] is CSS and the
two blocks share the same stabilizer/logical structure, transversal CX implements
**logical `CX̄` on all 3 logical-qubit pairs simultaneously**. The target block is
staged adjacent to the control block using `move_to` + `cz_partner` so the 8 pairs
are co-located for the entangling layer.

### 4.3 `logical_ccz(reg)` — the signature gate

`squin.broadcast.t(even-parity qubits)` and `squin.broadcast.t_adj(odd-parity
qubits)`, i.e. T on `{0,3,5,6}` and T† on `{1,2,4,7}`. This diagonal transversal
gate implements **logical `CCZ̄_{123}`** on the three logical qubits — the
defining transversal non-Clifford gate of [[8,3,2]].

**Claim to validate:** T-on-even / T†-on-odd realizes `CCZ̄` (orientation; the
opposite assignment yields the same gate since `CCZ` is Hermitian). Stated
explicitly in the docstring.

### 4.4 `logical_swap(reg)` — intra-block relabel via `permute`

Logical SWAP of logical qubits 1↔2 = swap of cube axes `x↔y`, i.e. the vertex
permutation `v=(x,y,z) → (y,x,z)`:

```
perm = [0, 1, 4, 5, 2, 3, 6, 7]    # qubits[i] -> current location of qubits[perm[i]]
```

(An involution; fixes the 4 vertices on the `x=y` diagonal, swaps `2↔4`, `3↔5`.)
This permutation maps `X̄_1↔X̄_2` and `Z̄_1↔Z̄_2` while preserving the stabilizer
group, so it is exactly **logical `SWAP̄_{12}`**.

Behavior under the toggle:
- `VIRTUAL = False` → `movement.permute(reg, perm)` (emit the physical move), then
  return the relabeled list `[reg[perm[i]] for i]`.
- `VIRTUAL = True` → return the relabeled list only; **no** physical move emitted.

Both yield the same logical effect; `VIRTUAL` toggles whether the relabel costs a
physical atom rearrangement or is a free software relabel.

**Implementation constraint (discovered):** a `movement.permute` that is the
*last* movement on a block does not concretise its placement — compilation fails
with `StaticPlacement body did not return a ConcreteState`. The swapped block
must therefore be consumed by a following operation, so `main` applies a gadget
to the swapped block after the swap (see §4.6). This only affects the physical
(`VIRTUAL = False`) path; the virtual path emits no permute.

### 4.5 `measure_logical_block(blocks)`

Mirrors `steane_demo.py`'s `measure_logical_reg`: the kernel allows only one
measurement, so flatten all blocks to physical qubits, issue a single
`squin.broadcast.measure`, and regroup the results into per-block (8-qubit)
slices for return.

### 4.6 `main()` + visualization

```
@physical.kernel(aggressive_unroll=True, verify=False)
def main():
    blocks = qalloc([0, 1])              # two [[8,3,2]] blocks, each in |000>_L
    a, b = blocks[0], blocks[1]
    transversal_cx(blocks[:1], blocks[1:])   # logical CX(a -> b)
    a = logical_swap(a)                       # logical SWAP_12 on block a
    logical_ccz(a)                            # consumes the swap (see §4.4) + CCZ on a
    logical_ccz(b)                            # logical CCZ on block b
    return measure_logical_block([a, b])      # flattened terminal measure
```

Then compile + visualize, matching the current demo's ending (using
`ASAPPlacePass` as the place optimiser, like `steane_demo.py`):

```
strat = make_physical_placement_strategy(return_moves=False, ...)
pipeline = PhysicalPipeline(placement_strategy=strat, place_opt_type=ASAPPlacePass)
compiled = pipeline.emit(main, no_raise=False)   # loud failure, not silent degradation
debugger(compiled, pipeline.arch_spec)
```

`no_raise=False` is important: the default `no_raise=True` silently emits a
degenerate empty program on a compilation error.

`verify=False` is required: the gadget kernels apply `ilist.map` over capturing
closures, which the verify pipeline cannot analyze (see
[[verify-nocloning-ilist-map-capture]]; kirin#679 / bloqade-circuit#830).

---

## 5. Testing / Verification

Per scope decision: **compile + visualize only**, no state simulation. Success
criteria:

1. `demo/move_demo.py` runs end-to-end (`uv run python demo/move_demo.py`) and
   produces the `debugger` visualization without raising. **Verified** for both
   `VIRTUAL = False` and `VIRTUAL = True` with `no_raise=False` (exit 0).
2. `pipeline.emit(main, no_raise=False)` produces a compiled move program with no
   residual `movement.*` user-directive statements (cz_partner resolved,
   move_to/permute lowered).
3. Flipping `VIRTUAL` changes the physical move count: **measured 63 `move.Move`
   statements with `VIRTUAL = False` (physical permute) vs 26 with
   `VIRTUAL = True`** (software relabel) — the virtual relabel more than halves
   physical atom movement.

(The demo is not part of the pytest suite; it is exercised via `just demo` /
direct run.)

---

## 6. Assumptions to Validate

1. **CCZ orientation** (§4.3): T-even / T†-odd → `CCZ̄_{123}`.
2. **GHZ schedule** (§4.1): the chosen H+CZ entangling schedule prepares GHZ =
   `|000⟩_L`; the schedule is move-friendly but not unique.
3. **Slot layout** (§4.1): concrete `(word, site)` assignment per slot is chosen
   for placement parallelism, not correctness; any valid distinct-address layout
   works.

---

## 7. Follow-Ups (Out of Scope)

- Optional fidelity/statevector verification of each gadget (would mirror
  `steane_demo.py`'s `FidelityAnalysis` path).
- A logical-CZ gadget (diagonal Clifford via `S/S†` on a chosen support) — omitted
  here to avoid an unverified support choice; can be added once the support is
  confirmed.
