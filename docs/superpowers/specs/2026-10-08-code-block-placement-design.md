# Code-Block-Aware Physical Placement — Design

- **Status:** Draft for review
- **Date:** 2026-10-08
- **Scope:** Physical pipeline (`PhysicalPipeline` → `PhysicalNativeToPlace` → `PlaceToMove`)

## Problem

A physical kernel that implements an error-correcting code groups its qubits
into code blocks — typically an `IList[Qubit, N]` per logical qubit, as in
`demo/steane_demo.py` and `demo/move_demo.py`. That grouping is lost during
compilation: after `SquinToNative` and `AggressiveUnroll` only individual qubit
allocations survive, and by `PlaceToMove` the layout heuristic and the placement
strategies see nothing but `qid → LocationAddress` maps, CZ stages and blocked
locations.

The default layout heuristic already favours "same site index across words",
which is what transversal gates between blocks need, but it cannot enforce it
because it does not know which qubits form a block.

## Goals

1. Let a kernel register physical qubits as members of a code block.
2. Carry block membership through inlining and unrolling to the layout
   heuristic and the placement strategies.
3. Make the default physical layout heuristic place each block as a contiguous,
   position-ordered run of sites in a single word.
4. Keep every existing program, layout and benchmark result unchanged when no
   blocks are registered.

## Non-goals

- Code blocks in the logical pipeline (`LogicalNativeToPlace`) or the generic
  `NativeToPlace`.
- Blocks that span several words. A block must fit in one word.
- Keeping blocks together after the initial layout. Placement strategies receive
  block membership and may use it, but no strategy is required to preserve the
  block shape across moves in this iteration.
- Block awareness in `PhysicalLayoutHeuristicFixed` or in the Rust solver.

## Rules

These are hard rules. Breaking any of them is an error, not a warning.

1. **Size.** A block has between 1 and `sites_per_word` qubits. A word may hold
   several blocks.
2. **Shape.** A block's initial layout is a contiguous run of sites in a single
   word, and position `p` sits at site `offset + p`.
3. **Pins are authoritative, and all-or-nothing.** Either every qubit in a block
   is pinned (with `gemini.common.new_at`, lowered from `NewAt`) or none is.
   - A fully pinned block must satisfy rules 1–2 through its pins alone; the
     compiler never moves a pinned qubit to repair a block.
   - A partly pinned block is an error.
4. **Membership.** A qubit belongs to at most one block. Qubits in no block are
   allowed and are placed as today.
5. **Placement must succeed.** If the layout heuristic cannot fit every unpinned
   block into the free sites of the home words, it raises
   `CodeBlockPlacementError`. Blocks are never silently dropped.

## Section 1 — Registration and lowering

### The `code_block` dialect

New dialect `bloqade.lanes.dialects.code_block` with one statement:

```python
@statement(dialect=dialect)
class Register(ir.Statement):
    """Declare `qubits` as one code block, positions in list order."""
    qubits: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType])
```

and a user-facing wrapper:

```python
from bloqade.lanes import code_block

@kernel
def main():
    a = squin.qalloc(7)
    b = squin.qalloc(7)
    code_block.register(a)
    code_block.register(b)
    ...
```

`register` has no runtime effect. Calling it inside an inlined helper (for
example a `new_steane_block()` allocator) is the intended pattern: every call
produces its own `Register` statement after inlining.

The dialect is added to the physical kernel dialect group. The logical
pipeline and the generic `NativeToPlace` delete `Register` statements and emit a
single `CodeBlockWarning` if any were present.

### `CodeBlockValidation`

A Kirin validation pass that runs in `PhysicalNativeToPlace._post_unroll_validation`,
after `AggressiveUnroll` and before `_lower_qubits`. At that point qubit
allocations are still `qubit.stmts.New` / `gemini.common.NewAt`, and pin
addresses are compile-time constants.

It reports **every** error in one pass, like the other validation suites:

| Error | Condition |
| --- | --- |
| `UnresolvedCodeBlock` | The `Register` argument is not a constant list of qubit allocations after unrolling (for example it is still inside control flow). |
| `EmptyCodeBlock` | The block has no qubits. |
| `DuplicateQubitInCodeBlock` | The same qubit appears twice in one block. |
| `QubitInMultipleCodeBlocks` | A qubit is registered in two blocks. |
| `CodeBlockTooLarge` | The block has more qubits than `arch_spec` has sites per word. |
| `PartiallyPinnedCodeBlock` | Some, but not all, members are pinned. |
| `PinnedCodeBlockShape` | A fully pinned block's pins span several words, are not contiguous, or do not satisfy `site = offset + position`. |

Under `no_raise=True` the pass follows the existing convention for the
post-unroll rules: if validation fails, no block is registered, all `Register`
statements are deleted, and one `CodeBlockWarning` lists the errors. A
silently-wrong block layout is worse than an unblocked one.

### `ResolveCodeBlocks`

A rewrite rule that runs in `PhysicalNativeToPlace._lower_qubits`, immediately
after `RewriteQubitsToPinnedQubits`:

- Block ids are assigned `0, 1, 2, …` in program order of the `Register`
  statements in the flattened kernel. The id is a stable identifier and the
  final tie-breaker in the layout heuristic. It does not choose the site offset.
- Each member's `place.NewPinnedQubit` gets
  `code_block = CodeBlockTag(block=<id>, position=<index in list>)`.
- The `Register` statements are then deleted.

`_NewQubitBase` gains `code_block: CodeBlockTag | None = info.attribute(default=None)`.
`NewLogicalQubit` never sets it.

### Opt-out

`PhysicalNativeToPlace` and `PhysicalPipeline` gain `use_code_blocks: bool = True`.
With `False`, `Register` statements are deleted without validation, one
`CodeBlockWarning` is emitted if any were present, and compilation is identical
to a kernel without registrations.

## Section 2 — Carrying blocks into analysis

### Types — `analysis/code_blocks.py`

```python
@dataclass(frozen=True)
class CodeBlockTag:
    block: int
    position: int

@dataclass(frozen=True)
class CodeBlock:
    block_id: int
    qids: tuple[int, ...]          # global qubit ids, in position order

class CodeBlockWarning(UserWarning): ...
class CodeBlockPlacementError(ValueError): ...
```

### Collection

`LayoutAnalysis` already collects `location_addresses` (the pins) from
`NewPinnedQubit`. It collects `code_block` tags the same way and groups them into
`code_blocks: tuple[CodeBlock, ...]`, sorted by `block_id`.

### Opt-in heuristic interface

```python
class CodeBlockLayoutHeuristicABC(LayoutHeuristicABC):
    @abc.abstractmethod
    def compute_layout_with_blocks(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, LocationAddress],
        code_blocks: tuple[CodeBlock, ...],
    ) -> tuple[LocationAddress, ...]:
        """Return a layout in which every block satisfies the shape rule, or raise
        CodeBlockPlacementError. Pinned qubits keep their pins."""
```

In `LayoutAnalysis.process_results`:

- No blocks → `compute_layout(...)`, exactly as today.
- Blocks, and the heuristic is a `CodeBlockLayoutHeuristicABC` →
  `compute_layout_with_blocks(...)`.
- Blocks, and the heuristic is not block-aware → one `CodeBlockWarning`
  naming the heuristic, then `compute_layout(...)`, and the blocks are not passed
  on. This is the backward-compatibility path.

After the layout is computed, `LayoutAnalysis` checks the shape rule on every
block as a post-condition. A block-aware heuristic that breaks it is a bug, so
this raises instead of warning.

### Into the placement strategies

`PlaceToMove.emit` passes the blocks that reached the heuristic to
`PlacementAnalysis` (default `()`).

`ConcreteState` gains:

```python
code_blocks: tuple[LocalCodeBlock, ...] = field(default=(), kw_only=True, compare=False)
```

```python
@dataclass(frozen=True)
class LocalCodeBlock:
    block_id: int
    members: tuple[int | None, ...]   # local index of position p, or None if that
                                      # qubit is not an argument of this state
```

- **Local index space.** Indices refer to `ConcreteState.layout`, which is what
  strategies already work in. `PlacementAnalysis.get_inintial_state` builds them
  from the global blocks and the state's qubit arguments. A block with no member
  in the state is omitted.
- **Out of equality and hashing** (`compare=False`), so lattice joins, caches
  and benchmark metrics are unaffected. `is_subseteq` is already field-explicit.
- **Re-attached around every strategy call.** Strategies build new
  `ConcreteState`s in many ways, and a direct constructor call drops the field.
  `PlacementAnalysis` sets `code_blocks` on the state it passes to
  `cz_placements` / `sq_placements` / `measure_placements` and on every state
  they return. Existing strategies need no change. Block-aware strategies can rely
  on the field being present, and on every block starting in one word on
  contiguous sites.

## Section 3 — The block-aware layout algorithm

`PhysicalLayoutHeuristicGraphPartitionCenterOut` (`heuristics/physical/layout.py`)
implements `CodeBlockLayoutHeuristicABC`. `compute_layout_with_blocks` runs the
same input checks as `compute_layout`. With an empty `code_blocks` it delegates to
today's `_compute_layout_from_cz_layers` unchanged.

Context on today's algorithm: `_compute_layout_from_cz_layers` builds candidate
slots by filling the first `k` home words from site 0 upward, then runs
`_global_site_min_cost_assignment` (greedy assignment and then pairwise-swap hill
climbing). Its cost is `_site_distance_matrix`, which scores site index (×100)
plus site-bus distance and ignores which word a qubit is in.

Fully pinned blocks reach this method as ordinary pins. Section 1 has already
checked them, so the algorithm only places **unpinned blocks** and **unblocked
qubits**.

1. **Capacity.** Pack items into the free sites of the home words (pinned sites
   are occupied), first-fit in decreasing size: each unpinned block is an item of
   its size that needs a contiguous free run in one word, and each unblocked
   qubit is an item of size 1. `k = max(_word_count(len(qubits)), words used by
   the packing)`. If the packing needs more home words than exist, raise
   `CodeBlockPlacementError` naming the first block that did not fit.
2. **Rank blocks by entanglement weight.** A block's weight is the total CZ count,
   over all stages, on edges with exactly one endpoint in the block. Internal
   edges are left out because the block's fixed shape already settles them. Order
   by weight descending, then size descending, then `block_id` ascending.
3. **Greedy block placement.** In rank order, give each block the feasible
   `(word, offset)` slot among the first `k` home words that adds the least cost
   against pinned qubits and already-placed blocks. Cost is
   `Σ w(a, b) · site_distance[site(a)][site(b)]` over CZ edges, with
   `site = offset + position`. Ties go to the lowest offset, then the lowest word.
   Two blocks linked by transversal CZs want equal offsets, so they end up in
   different words.
4. **Block hill climbing.** Deterministic passes over block pairs and slots: swap
   two blocks of equal size, or move a block to a free feasible slot. Accept only
   strict cost decreases, with at most `len(blocks)` passes, mirroring the existing
   swap loop. Block shape is preserved.
5. **Unblocked qubits.** Run `_global_site_min_cost_assignment` over the free sites
   left in the first `k` words, including leftover sites in block words. Placed
   block qubits act as **fixed anchors**: an unblocked qubit's incremental and swap
   cost include its CZ edges to block qubits, so syndrome-style qubits are pulled
   toward the site indices of the block qubits they interact with. Edges to
   pinned-but-unblocked qubits stay excluded, as today. This needs an optional
   `anchors: dict[int, LocationAddress]` parameter on
   `_global_site_min_cost_assignment`; when the parameter is empty the function
   behaves exactly as now.

### Invariant

No registered blocks, or `use_code_blocks=False`, or a heuristic that is not
block-aware, gives a layout bit-identical to today's. The existing benchmark rows
in `latest_physical.csv` / `latest_logical.csv` must not change.

`PhysicalLayoutHeuristicFixed` stays non-aware and takes the warning path.

## Section 4 — Tests

**Registration and validation (Section 1)** — `python/tests/rewrite/`, `python/tests/validation/`

- An inlined allocator that registers a block, called twice, gives two distinct
  blocks with correct positions after unrolling.
- One kernel with an oversized block, a partly pinned block, non-contiguous
  pins, mis-ordered pins and pins in two words reports all errors in one pass.
- `no_raise=True` with invalid blocks: no tags, one `CodeBlockWarning`.
- `use_code_blocks=False`: IR and layout identical to the same kernel without
  `register` calls, plus one warning.
- The logical pipeline deletes `Register` with one warning.

**Analysis plumbing (Section 2)** — `python/tests/analysis/layout/`, `python/tests/analysis/placement/`

- `ConcreteState.code_blocks` survives every built-in strategy (parametrized over
  the strategy registry) and does not affect equality or hashing.
- Local index translation, including blocks only partly present in a state.
- A non-block-aware heuristic (`PhysicalLayoutHeuristicFixed`) emits one warning
  and gives the same layout as without blocks.
- The layout post-condition raises on a heuristic that breaks the shape rule.

**Layout algorithm (Section 3)** — `python/tests/heuristics/test_physical_layout_code_blocks.py`

- Every block lands in one word, contiguous, `site = offset + position`.
- Two blocks with transversal CZs get the same offset in different words.
- Several blocks share a word when that is cheaper.
- More-entangled blocks are placed first: a small case where the order changes
  the result.
- Unblocked qubits are pulled toward the block qubits they interact with
  (anchors).
- Fully pinned blocks stay exactly at their pins.
- `CodeBlockPlacementError` when the blocks cannot be packed.
- No blocks: layouts identical to today's on the existing pinned and unpinned
  test kernels.

**End to end** — `python/tests/integration/`

- A Steane-style kernel with two 7-qubit blocks and a transversal CX compiles
  through `PhysicalPipeline`. The initial layout respects both blocks, and the
  transversal CZ layer becomes identical site moves across the two words.

**Benchmark (optional, decide at review)**

- A new small case in `python/benchmarks/kernels/` that registers blocks, with new
  rows in `latest_physical.csv`. Existing rows must not change.

## Open questions for review

1. Should `register` in the logical or generic pipeline warn (as proposed) or
   error?
2. Under `no_raise=True`, is "drop all blocks with one warning" the right fallback?
3. Include the benchmark case in this change or in a follow-up?
