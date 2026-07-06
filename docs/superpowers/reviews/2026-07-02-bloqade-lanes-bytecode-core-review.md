# Crate Review: bloqade-lanes-bytecode-core (2026-07-02)

Focus: the crate's integration with vihaco (`vihaco` / `vihaco-cpu` / `vihaco-parser`), following the #769 migration.

> **Resolution (2026-07-02, commit `24970e7`).** In-scope, low-risk cleanups on this PR's own surface were fixed: **`validate_all` removed** (dead + wrongly omitted `simulate_stack`; its coverage folded into a `validate_structure`+`validate` chain test), and **`TextError` moved to `isa::text`** where it's produced (restoring cohesion). Deferred — mostly the `NewArray`/array-refactor blast zone (#776) or larger surface changes: the `isa/validate.rs` simulator split (§4), the `Program` newtype firewall (§3.1), the no-arch duplicate-check fallbacks (§4 pattern 1), `op_name`-from-`#[token]` (guarded by test), the `Instruction::Cpu` leakage / vihaco-semver question, the error-umbrella, and `BadInstruction.line: 0` (needs upstream vihaco spans). The open questions below stand as-is for those.

## 1. Context

`bloqade-lanes-bytecode-core` is the pure-Rust foundation of the Bloqade Lanes SDK: it owns the architecture spec (`arch`), the atom-movement simulator (`atom_state`), versioning (`version`), and — the subject of this review — the bytecode instruction set, program container, text/binary codecs, and validation (`isa`). It depends on `chumsky`, `eyre`, `serde`/`serde_json`, `thiserror`, and the four vihaco crates. Four workspace crates consume it: `bloqade-lanes-bytecode-python` (PyO3 bindings), `bloqade-lanes-bytecode-cli` (CLI + C FFI), `bloqade-lanes-dsl-core`, and `bloqade-lanes-search`.

Change activity is **heavy and almost entirely AI-authored**: ~21 commits in the last 30 days, all part of the vihaco migration (#769) and the follow-on PR-review work (#544/#547/#770). The `isa` module was rebuilt on vihaco: `Instruction` derives vihaco's binary codec + parser, `Program` is a type alias over `vihaco::module::Module`, CPU ops are delegated to `vihaco_cpu::Instruction`, text parsing uses vihaco's `ParsedModule`, and the legacy hand-rolled `bytecode` module was deleted. Current `isa/` line counts: `validate.rs` 1050, `def.rs` 422, `program.rs` 278, `text.rs` 242, `parse_helpers.rs` 32, `mod.rs` 45.

## 2. External API Surface

*(Agent 1 — sampled top ~9 of 11 consumer files by pub-type match density.)*

### Public Type Inventory (highlights)
| Item | Kind | Module | Consumers |
|------|------|--------|-----------|
| `Instruction` | enum (lanes-native variants + `Cpu(vihaco_cpu::Instruction)`) | `isa::def` (re-exported `isa`) | python, cli, dsl-core |
| `INSTRUCTION_WIDTH` | `const u32 = 17` | `isa::def` | cli, python (indirect) |
| `Program` | `type = Module<Instruction, Value, Type, LanesInfo>` | `isa::program` | python, cli, ffi, search |
| `LanesInfo` | struct `{ version: Version }` | `isa::program` | python, cli |
| `from_code` | `fn(Version, Vec<Instruction>) -> Program` | `isa::program` | python, cli |
| `to_binary`/`from_binary` | fns | `isa::program` | python, cli, ffi |
| `BinaryError` / `TextError` | enums (4 / 3) | `isa::program` | python, ffi |
| `parse_text` / `to_text` | fns | `isa::text` | python, cli, ffi |
| `ValidationError` | enum (20, incl. stack-sim) | `isa::validate` | python, cli, ffi |
| `validate` / `validate_structure` / `validate_all` / `simulate_stack` | fns | `isa::validate` | python, cli |
| `tag` | submodule of `u8` consts | `isa::validate` | internal + Python numeric parity |
| `ArchSpec`, `Word`, `Grid`, `Bus<T>`, `Zone`, `Mode`, `TransportPath` | topology | `arch::types` | all four |
| `LocationAddr`, `LaneAddr`, `ZoneAddr`, `Direction`, `MoveType` | addr codec | `arch::addr` | all four |
| `AtomStateData` | struct (5 pub `HashMap`) | `atom_state` | python, search |
| `Version` | struct | `version` | everywhere |

**Vihaco types leaking through the public API:** `Program` structurally exposes `vihaco::module::Module` + `vihaco::value::{Value,Type}` (consumers read `.code` / `.extra.version` directly); `Instruction::Cpu(vihaco_cpu::Instruction)` forces `vihaco_cpu` onto exhaustive matchers; the Python binding imports `vihaco::instruction::OpCode` and `vihaco::value::Value` directly.

### Responsibility Portraits (summary)
- **`isa`** — the heart of the migration; four jobs: ISA def (`def.rs`), program container (`program.rs`), text codec (`text.rs`), validation (`validate.rs`). Python/CLI/FFI are thin wrappers over it.
- **`arch`** — the topology oracle: device description + address codec + a rich query surface all four consumers lean on.
- **`atom_state`** — immutable qubit↔location move simulator; delegates endpoint/CZ resolution to `ArchSpec`.
- **`version`** — cross-cutting `Version` used in arch spec, program header, and binary packing.

### API Friction Points
1. `Program` is a **type alias, not a newtype** — no compilation firewall against `Module` field changes.
2. `Instruction::Cpu` leaks `vihaco_cpu` to exhaustive matchers.
3. `TextError::BadInstruction { line: 0 }` — line number always zero (documented regression via `ParsedModule`).
4. Three disjoint error enums (`Binary`/`Text`/`Validation`), no umbrella.
5. `AtomStateData` exposes five `pub HashMap` fields; the forward/reverse sync invariant is unenforced.
6. `validate_all` is defined but not re-exported at `isa::` level and unused (CLI/Python chain manually).
7. `INSTRUCTION_WIDTH = 17` is a documented future break (#776).

### Dead Public Surface
`validate_all` (no consumer), `MAGIC` (tests only), `isa::parse_helpers` (only via `#[parse_with]`; could be `pub(crate)`), `tag` constants (pub only for Python numeric parity), some `Bus::resolve_*` / `TransportPath::check_finite`.

## 3. Internal Architecture

*(Agent 2.)*

### Module Map
`lib` (re-exports) → `version`; `arch::{addr → types → validate/query → mod}`; `atom_state`; `isa::{def, parse_helpers, program, text, validate, mod}`.

### Internal Interaction Graph (key edges)
- `version` → `arch::types`, `isa::program`, `isa::text` (`Version`)
- `arch::addr` → `arch::{types, query, validate}`, `atom_state`, `isa::validate` (address types)
- `arch::types` → `arch::{validate, query}`, `atom_state`, `isa::validate` (`ArchSpec`)
- `arch::query` → `isa::validate` (`LocationGroupError`, `LaneGroupError`)
- `isa::def` → `isa::{program, text, validate}` (`Instruction`, `INSTRUCTION_WIDTH`); `isa::parse_helpers` → `isa::def` (via `#[parse_with]`)
- `isa::program` → `isa::text`, `isa::validate` (`Program`, `from_code`, `TextError`)

### Responsibility Portraits (internal)
- **`ArchSpec`** — data in `arch::types`, behavior split across `arch::validate` (`validate()`) and `arch::query` (all lookups); the type is unaware of the file split.
- **`Instruction`** — fully proc-macro-derived codec + parser; `Cpu(vihaco_cpu::Instruction)` `#[delegate]` splices in vihaco-cpu as a nested sub-ISA; hand-written `Display`/`op_name` complete it.
- **`Program`** — transparent `Module` alias; no methods, all free fns; `Module.functions` populated by parse then discarded (resolver keeps `functions[0].body`).
- **`AtomStateData`** — immutable value type; self-contained, only imports `arch`.
- **`StackSimulator`/`SimEntry`** (private) — single-pass abstract interpreter in `validate.rs`.
- **`ValidationError` + `tag`** — one enum spanning capability/address/structural/stack-sim; `tag` is a cross-language contract with the Python runtime.

### Internal Coupling Hotspots
- **`isa::validate` — 5 sibling imports, 1050 lines, four co-resident concerns** (capability ~50, address ~30, structural ~60, stack simulator ~300 + ~450 test lines). Decomposition opportunity: move `StackSimulator`+`SimEntry`+`tag` to `isa::simulate`/`isa::stack` — no public-API change.
- `arch::query` — secondary hotspot (3 arch siblings; justified).
- `atom_state` — clean (2 imports, no ISA knowledge).

## 4. Critical Evaluation

*(Agent 3 — hotspot files read directly.)*

### Contract vs Implementation Divergence
| Public type / contract | Verdict | Evidence |
|---|---|---|
| `Program` "vihaco `Module` specialised… single `@main`" | **GAP-drift** | Alias not newtype; `Module.functions` discarded; `.code`/`.extra` public; "single `@main`" enforced only at parse time. |
| `from_code` "one constructor, source-independent equality" | **MATCHES** | Both `from_binary` and resolver funnel through it; round-trip tests confirm. |
| `to_binary` layout | **MATCHES** | Matches module doc; one justified `expect`. |
| `TextError::BadInstruction{line}` | **GAP-communication** | `line` always `0`; documented at use site but the field advertises a value it never delivers. |
| `TextError::InvalidVersion` | **GAP-drift** | Documented unreachable; dead variant (malformed version routes to `BadInstruction`). |
| `validate`/`validate_structure`/`simulate_stack` | **MATCHES** | Collect-all, `None` skips arch; verified by tests. |
| `validate_all` | **GAP-drift** | Defined + tested but wired to nothing; also omits `simulate_stack`, so not even the composition consumers want. |
| `Instruction::op_name` | **MATCHES** | Pinned by tests; consumed by Python decoder. |
| `INSTRUCTION_WIDTH = 17` | **MATCHES (documented future break)** | Width test passes; #776 deferral documented. |

### Rust Health Findings (hotspot files)
- `program.rs:143` `to_binary` `expect("writing … to a Vec cannot fail")` — **justified** (`Vec`'s `Write` is infallible; message names the invariant). No action.
- No `unwrap`/`panic`/`todo!`/`unimplemented!` in non-test code across the five hotspot files; every `unwrap` is under `#[cfg(test)]`.
- No `unsafe` in core hotspots (FFI `unsafe` lives in the CLI crate, out of scope).
- Lifetimes healthy — only `StackSimulator<'a>` (one borrowed `Option<&ArchSpec>`); no creep.
- `#[allow(clippy::field_reassign_with_default)]` on `from_code` — correctly reasoned (foreign `Module`, no struct-literal init).
- `MAX_TYPE_TAG` boundary uses `>` (not `>=`), correctly keeping `0x8` valid and `0x9`/measurement-result out — consistent with #770.

### Architectural Health Findings
- **`isa::validate.rs` 1050 lines / four concerns** — the crate's single biggest comprehension tax; the stack simulator shares only `ValidationError` + arch `check_*` with the other validators. Clean split available with zero public-API change.
- **`TextError` defined in `program.rs` but produced only in `text.rs`** — half-cohesive module (`BinaryError` *is* produced in `program.rs`); this split is why the always-zero `line` is easy to overlook.
- **Vihaco leakage, three vectors, no compilation firewall** — `Program` alias, `Instruction::Cpu`, and transitive re-export of `vihaco::{instruction,value}` to downstream crates. Deliberate and documented (`mod.rs:20-35`), with real payoff (derived codec + parser), but crate stability is transitively bounded by vihaco semver and that is invisible at the `pub use` surface.
- **`arch::query` split** — group-validation `check_*` + `DuplicateAddress` variants live here while `isa::validate` reaches across; justified but arch-validation logic spans two modules.

### AI-Drift Findings
The whole window is AI-authored (~20 commits). Stylistic coherence is high (consistent doc density, `pc`-tagged errors, collect-all discipline). Drift shows up as **unwired surface + reconciliation churn**, not sloppiness:
- Dead/unwired: `validate_all` (never called), `TextError::InvalidVersion` (self-documented unreachable), `MAGIC` (tests only).
- **Reconciliation churn as its own commit stream** — `e8b4f4e`, `70e4311`, `6a7b8ea`, `72b6218` are after-the-fact fixups of first-pass AI review debt (the `line:0` and dead `InvalidVersion` are residue).
- The `op_name`/`Display`/`#[token]` triple re-enumerates all 20 variants three times; mitigated by a pinning test (a test, not a single source).
- **Good news:** no duplicated logic across the vihaco boundary — genuine delegation (`#[delegate]`, `Display` passthrough), the healthy version of this integration.

### ⚠ Emerging Patterns

⚠ Emerging Pattern: "No-arch duplicate-only fallback shadows the real validator"
  Appears in: `isa/validate.rs:434-462` (`check_duplicate_locations` / `check_duplicate_lanes`) vs `:473,:491` (`arch.check_locations`/`check_lanes`)
  Similarity: two near-identical private methods re-implementing only the duplicate-address subset of arch validation, differing only in `encode()` shape (`u64` vs `(u32,u32)`), with their own `HashSet<seen>`/`HashSet<reported>` bookkeeping
  Signal: 2 instances, last touched in the simulator port (~1–3 days ago)
  Suggested abstraction: drop no-arch group checks entirely, or a shared `dedup_addresses<T: Eq+Hash>(iter) -> Vec<Dup>` helper
  Readiness: still evolving

⚠ Emerging Pattern: "Three parallel error families with no umbrella"
  Appears in: `isa/program.rs` (`BinaryError`, `TextError`), `isa/validate.rs` (`ValidationError`), + three matching `*_to_py` conversion families in the python crate
  Similarity: disjoint enums sharing only `std::error::Error`; each new failure mode picks one enum and threads a fourth conversion path
  Signal: 3 enums, structural across the whole integration
  Suggested abstraction: an `IsaError` umbrella enum (or trait) unifying the three
  Readiness: monitor

⚠ Emerging Pattern: "Variant-count fan-out (N match arms per instruction)"
  Appears in: `isa/def.rs` `op_name` (:128), `Display` (:170), `isa/validate.rs` `StackSimulator::dispatch`, + `control_flow_mnemonic`/`cpu_op_name`
  Similarity: each `Instruction` variant hand-maintained in ≥4 exhaustive matches (the derive collapses only codec+parser)
  Signal: 4+ match sites, load-bearing as the ISA grows toward the #776 array refactor
  Suggested abstraction: derive `op_name` from `#[token]` metadata; table-drive the semantic arms where possible
  Readiness: monitor

⚠ Emerging Pattern: "Documented-deferral-as-load-bearing-comment"
  Appears in: `isa/def.rs:25` (`INSTRUCTION_WIDTH`/#776), `isa/program.rs` (`InvalidVersion` "retained for API stability", `BadInstruction.line: 0`), `isa/validate.rs:273` (`MAX_TYPE_TAG` boundary)
  Similarity: four spots where a comment carries a decision the type/value can't express
  Signal: 4 instances; good discipline but concentrates correctness in prose
  Readiness: emerging (monitor)

## 5. Open Questions

### Contract Divergence
1. Should `Program` become a **newtype** wrapping `Module` (with `.code()`/`.version()` accessors) to establish a compilation firewall — given `Module.functions` is already discarded and the "single `@main`" invariant is unenforced by the alias? What breaks in CLI/Python if `.code`/`.extra` stop being public fields?
2. Should `TextError::InvalidVersion` be **deleted** (provably unreachable today), or is a code path planned that will reach it?
3. Is `validate_all` meant to be the canonical entry point (then why does it omit `simulate_stack`, and why do both consumers bypass it), or should it be removed?

### Rust Health
1. `to_binary`'s `expect` is the only panic surface in the hotspot set and is justified — is there any appetite to make serialization total (return `Result`) for consistency with `from_binary`, or is the infallible-`Vec` `expect` the preferred idiom to keep callers clean?

### Architectural Health
1. Can `StackSimulator` + `SimEntry` + `pub mod tag` move to `isa::simulate`/`isa::stack` (with `ValidationError` staying shared) to cut `validate.rs` from 1050 → ~600 lines with zero public-API change? Where should `ValidationError` live to avoid a circular module dep?
2. Should `TextError` move from `program.rs` to `text.rs` (where it is produced) to restore module cohesion and make the `line: 0` regression visible at its definition?
3. Is `Instruction::Cpu(vihaco_cpu::Instruction)` intended as **stable public surface** or an implementation detail? If the latter, does the exhaustive-match leakage onto downstream crates argue for a lanes-owned CPU-op enum that converts at the boundary? And what is the assumed vihaco semver/stability contract, given crate stability is transitively bounded by it?

### AI-Drift
1. The `#770`-review churn cluster (`e8b4f4e`/`70e4311`/`6a7b8ea`/`72b6218`) fixed diagnostics/values that the first migration pass shipped mismatched. Is a lightweight "diagnostic message ↔ behavior" check (or snapshot test of error `Display`) worth adding so these don't recur?
2. Should the always-zero `TextError::BadInstruction.line` be fixed by recovering real spans from vihaco's `ParsedModule` (does vihaco expose per-instruction spans?), or should the `line` field be dropped rather than advertise a value it never provides?

### Emerging Patterns
1. Should the no-arch `check_duplicate_locations`/`check_duplicate_lanes` fallbacks be **deleted** in favor of a "no-arch ⇒ no group checks" policy, or unified with `arch::query`'s duplicate logic so the two can't diverge? Is duplicate-address detection even meaningful without an arch spec to define what an address *is*?
2. Is an `IsaError` umbrella (unifying `BinaryError`/`TextError`/`ValidationError`) worth it, or does the three-way split reflect genuinely different call sites that shouldn't be merged?
3. Can `op_name` be derived from the `#[token]` metadata the `Parse` derive already consumes, removing the third hand-maintained match arm — or is the deliberate CPU-op divergence (`const.f64` text vs `const_float` handler name) fundamental enough that the pinning test is the right long-term guard as the ISA grows toward #776?
