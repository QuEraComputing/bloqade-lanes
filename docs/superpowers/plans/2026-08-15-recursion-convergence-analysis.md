# Prototype: separating convergent from divergent recursion

Prompt for a fresh session. Exploratory — the first deliverable is a written
judgement on feasibility, not code.

## Where things stand

PR #923 added a guard in `gemini/common/validation/recursion.py`, run from
`logical.kernel`'s `run_pass`, that rejects a kernel whose call graph is cyclic
or not statically resolvable. It exists because a cyclic call graph did not
error, it **hung**: `AddressAnalysis` re-analyses each callee at every call site
with no memoisation, and kirin's depth guard returns bottom instead of raising,
so a branching cycle costs `phi ** max_depth` interpreter calls (measured: 1.618
per unit depth, ~10^167 at the default 800).

Two rules ship today, both deliberately blunt:

- `NoRecursionValidation` — any cycle in the statically resolvable call graph.
- `NoOpaqueCallValidation` — any `func.Call` whose callee is one of the kernel's
  own parameters, scanned across every reachable method.

The second is a **shape rejection**, not a detection. A call through a parameter
is refused because resolving it would need the constant propagation that
diverges on the very input being guarded against. That closes the hang, but it
also rejects every higher-order kernel, convergent or not. Read the module
docstring and `docs`-linked issues before starting; the `Cycle`/`route`
machinery and the `%<name>_self` handling both encode findings that are easy to
rediscover the hard way.

## The goal

Replace shape rejection with a real decision. Given a kernel, classify its
recursion as:

1. **Divergent** — cannot converge; reject with a precise reason.
2. **Convergent and compile-time bounded** — accept, with a known unrolling.
3. **Cannot prove either way within budget** — reject, but say so honestly and
   distinguish it from (1) in the message.

Three outcomes, not two. Collapsing (3) into (1) produces confident wrong
diagnostics; collapsing it into (2) reintroduces the hang.

## Why this is not quite the halting problem

Deciding termination in general is undecidable, so if the question were "does
this kernel terminate", the answer would be "no, go home". It isn't, and the
reason is specific to this compiler:

**A Gemini kernel must lower to a fixed sequence of physical atom moves.** There
is no runtime control flow on the device. So the acceptance criterion is not
"terminates" but "**can be fully unrolled at compile time into a bounded,
statically known sequence**" — which is strictly stronger, and decidable
relative to a budget.

That reframing does the real work:

- A recursion is acceptable only if every value controlling its depth is a
  compile-time constant. That is a question about the constant-propagation
  lattice, not about halting.
- Anything whose depth depends on a runtime value is rejected regardless of
  whether it terminates — it cannot be lowered either way.
- The budget makes "cannot prove" a legitimate, sound answer rather than a gap.

So the honest target is a **sound accepter**: never accept something that cannot
be unrolled; freely reject things that could in principle converge but were not
proved within budget. False rejects are compile errors telling the user to make
the depth static, which is a reasonable thing to demand of a kernel language.

Separately, a **sound "definitely divergent" detector** is worth having purely
for error quality — `f(n) = f(n)` should say "this can never converge", not "I
gave up". See size-change termination below; that is exactly what it decides.

## Prior art worth reading first

In-repo:

- `bloqade/analysis/address/{analysis,impls}.py` — the negative example. No
  memoisation, and the lattice grows (fresh addresses), so it re-walks every
  path. Understand precisely why it diverges before designing a replacement.
- `kirin/analysis/const/prop.py` — the existing constant propagation, and the
  thing that would decide whether a recursion depth is static.
- `kirin/interp/abc.py` — `max_depth`, and `recursion_limit_reached` returning
  bottom rather than raising (`interp/abstract.py:79`).
- `bloqade/rewrite/passes/aggressive_unroll.py` and kirin's `Call2Invoke` —
  bounded unrolling already exists for loops. Ask whether recursion can reuse it.

Literature:

- **Size-change termination** (Lee, Jones, Ben-Amram, POPL 2001). Build a
  size-change graph per call edge recording which parameters strictly decrease;
  the program terminates if every idempotent composition in the closure has a
  strictly decreasing self-loop. Decidable, PSPACE-complete. This is the
  canonical answer to "can I decide termination for a restricted but useful
  class", and it handles mutual recursion natively.
- **Ranking function synthesis** — linear (Podelski & Rybalchenko 2004) and
  lexicographic variants; transition invariants / Terminator (Cook et al.).
- **0-CFA / k-CFA** (Shivers) for call-graph construction in the presence of
  first-class functions. 0-CFA is very likely enough here.
- **Abstract interpretation with widening** (Cousot & Cousot) if any domain used
  has infinite height.

## Suggested staging

Each stage is independently useful; stop and report after any of them.

**Stage 1 — call-target analysis.** A `Forward` analysis whose lattice is "the
set of methods this SSA value may hold". Finite (bounded by the methods in the
program) and monotone, so a fixpoint is guaranteed — which is exactly why it
terminates where `AddressAnalysis` does not. Memoise on `(method, abstract
inputs)`; that memo table is the missing ingredient. Needs method tables for
whatever moves method values around: `func`, `py.tuple`, `ilist`, plus
`PartialLambda`.

Deliverable: a call graph with dynamic edges resolved, feeding the existing
cycle DFS. On its own this replaces `NoOpaqueCallValidation`'s blanket rejection
with real detection, and lets genuinely acyclic higher-order kernels through.

**Stage 2 — convergence decision on each cyclic SCC.** For each cycle Stage 1
finds, decide divergent vs bounded. Start with the cheap, high-value cases:

- self-call with abstract arguments unchanged on every path → definitely
  divergent (this is the degenerate size-change case and covers the reported
  bug);
- depth controlled entirely by constants that const-prop can evaluate → bounded,
  and the bound falls out;
- anything else → cannot prove within budget.

Only reach for full SCT if the cheap cases prove insufficient on the corpus.

**Stage 3 — act on it.** Unroll the bounded ones, or hand the bound to the
existing unroller. Possibly the whole thing collapses into "let the inliner
recurse with a budget, driven by const-prop base-case detection" — evaluate that
cheaper alternative explicitly before building a new analysis, and say which is
better and why.

## Corpus and expected verdicts

Build these first and keep them as the yardstick. Several are in
`python/tests/gemini/validation/test_recursion_validation.py` already.

| kernel | verdict |
|---|---|
| `def f(n): return f(n)` | divergent — argument never changes |
| `main()` calling itself, result discarded (the reported bug) | divergent |
| `main -> B`, `B -> {B, main}` (the `#921` mwe) | divergent |
| `B(f): f(); f()` with `main` passed in | divergent — dynamic, branching 2 |
| `def f(n): return 0 if n == 0 else f(n - 1)`, called `f(5)` | bounded, depth 5 |
| same, called `f(x)` with `x` runtime | cannot prove — depth not static |
| same, called `f(10_000)` | cannot prove within budget (or bounded, if the budget allows — decide and document which) |
| acyclic higher-order kernel, parameter called but no cycle | accept (today's rule wrongly rejects this) |
| mutual recursion with a measure decreasing across two kernels | bounded in principle; note that kirin cannot currently express it (forward references do not lower) |

## Hard constraints

- **The analysis must terminate on every input, including the ones that hang
  today.** Prove it from the lattice (finite height + monotone + memoised), not
  by testing. A depth or iteration cap is a backstop, not the argument.
- No wall-clock regression on the existing suite. Time the guard before/after.
- Deterministic output — sort by `(sym_name, id)` as the current code does;
  diagnostics are compared in tests.
- Keep the existing guard working throughout. It is shipped and it fixed a real
  user hang; this replaces it only once the replacement is demonstrably sound.

## Deliverables

1. A written feasibility judgement, first, before implementation: which of the
   three verdicts is decidable here, at what cost, and what the soundness
   direction is. If the honest conclusion is "keep the shape rejection", say so
   and explain what would have to change to make it worth revisiting.
2. If it looks feasible: a Stage 1 prototype behind a flag, off by default, with
   the corpus above as tests and a termination argument in the module docstring.
3. A note on what it would take to lift `NoOpaqueCallValidation` — that is the
   user-visible payoff, since it is what currently rejects higher-order kernels.

## Ground rules

Verify claims against the code and by running things; several facts in this area
are counterintuitive and were established the hard way (a self-call is a
*dynamic* call on `%<name>_self` at guard time; `ir.Method.backedges` is empty
during `run_pass` and stale afterwards; `Call2Invoke` needs const hints that only
exist after the pass that diverges). Do not accept any of those secondhand —
including from this document. Report negative results plainly; "this cannot be
done soundly for reason X" is a successful outcome here.
