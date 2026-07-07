# Adopt vihaco's `Module` + text parser for the bytecode ISA

**Status:** approved (design)
**Date:** 2026-07-01
**Context:** PR #770 review feedback (issues #769, #544, #547; upstream vihaco#34)

## Goal

Address the PR #770 review by leaning further into vihaco's own infrastructure:

1. Use vihaco's `Module` as the in-memory program object instead of our bespoke
   `Program` struct.
2. Use vihaco's built-in **text parser** (`ParsedModule` → `Resolve` → `Module`)
   for reading, replacing the hand-rolled line-driver `parse_text`.
3. Keep a **hand-rolled binary** codec (the `LANES` container), but operating on
   `Module`.
4. Roll a small round-trippable **text emitter** matching vihaco's parse grammar
   (plan A), since vihaco's `Module::Display` is a debug view, not re-parseable
   (tracked upstream as vihaco#34).
5. Fold in the rest of the review feedback: rename `vihaco_isa` → `isa`, add the
   measurement-result value tag (#547), and fix the five stale-diagnostic nits
   Copilot flagged.

This also closes #544 (mnemonic-string drift): routing per-instruction printing
through each instruction's `Display` makes the derive `#[token]` the single
source of truth for the token, consumed by parser, emitter, and `op_name()`.

## Non-goals

- The array / measurement-result → nested-heap-`IList` rework (#776) — separate.
- A round-trippable emitter *in vihaco* (vihaco#34) — we ship the local emitter
  until that lands.
- Multi-function / multi-section programs — we remain single-function.

## Design

### Program object: `Module`

Replace `isa::Program` with vihaco's
`Module<Instruction, Value, Type, LanesInfo>`:

- `code: Vec<Instruction>` — the instruction stream (what we care about).
- `functions`, `labels`, `constants`, `strings`, `source_symbols`,
  `main_function` — a single `@main` function spanning the whole code; the rest
  empty (we carry no constants pool / string interner / labels).
- `extra: LanesInfo` — a small consumer-metadata struct holding the program
  `version` (`Module` has no version field of its own).

```rust
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LanesInfo { pub version: Version }
// impl Display for LanesInfo (Module: Display requires Info: Display)

pub type Program = Module<Instruction, Value, Type, LanesInfo>;
```

Keeping the `Program` alias limits churn in consumers and expresses intent.

### Text format (visible, breaking within this PR's new format)

Adopting vihaco's parser changes the source grammar from the flat `.sst` to
vihaco's header-lines + `fn @main { … }`:

```
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  const_loc 0x0000000001000000
  initial_fill 2
  const.f64 1.5708
  global_rz
  measure 1
  await_measure
  halt
}
```

- `version 1.0;` is a **header line** parsed by a `LanesHeader` type
  (`H` in `ParsedModule<I, H>`); the resolver folds it into `LanesInfo.version`.
- Instruction lines inside the body are unchanged from the current ISA
  (`const_loc`, `move`, `const.f64`, …).

### Reading: `ParsedModule` + `Resolve`

`Program::parse_text(src)`:

1. `ParsedModule::<Instruction, LanesHeader>::parser().parse(src)`.
2. A `Resolve` impl builds the `Module`: take the single `@main` function's body
   (all `BodyItem::Direct`, since every lanes instruction parses directly — a
   `Raw` form is an error, which the default `resolve_body` already reports),
   collect into `code`, set `main_function`, and read the version header into
   `extra`.

Errors map to the existing `TextError` variants (missing/invalid version, bad
instruction), converted to Python exceptions as today.

### Writing: hand-rolled round-trippable emitter (plan A)

`Program::to_text()` emits the exact parse grammar:

```
version {major}.{minor};
fn @main() {
  {inst}      # one per line, via <Instruction as Display>
  ...
}
```

Per-instruction rendering goes through a **`Display` impl on `Instruction`**
(reusing `vihaco_cpu::Instruction`'s `Display` for the `Cpu(..)` arm). This
replaces the hand-written `instruction_to_text` match and the separate `repr`
match, and is the single source of truth for the printed mnemonic — closing
#544. A round-trip test (`parse_text(to_text(p)) == p`) guards emitter↔parser
agreement.

### Binary: `LANES` codec over `Module`

Unchanged in spirit; `to_binary`/`from_binary` now read/write a `Module`:

```
magic   : 5 bytes  = b"LANES"
version : u32 LE   = (major << 16) | minor    # from module.extra.version
code    : N × INSTRUCTION_WIDTH bytes          # module.code, vihaco WriteBytes/FromBytes
```

`from_binary` builds a `Module` with a single `@main` function and
`extra.version`. No vihaco module-level binary format exists, so this stays ours
(`BinaryError` unchanged).

### Validation

`validate_structure` / `validate` / `simulate_stack` keep taking `&Program`
(now the `Module` alias) and iterate `program.code` internally; logic is
otherwise unchanged. `Program::validate` / `validate_all` stay as methods. Arch
capability checks that read the version (none today) would use `program.extra`.

### Rename

`vihaco_isa` module → `isa` throughout (core `lib.rs`, CLI, FFI, PyO3, tests).

### Review-comment fixes folded in

- Add `tag::MEASUREMENT_RESULT = 0x9` to the value tags (#547); wire
  `await_measure`'s pushed value to it in the simulator (measurement result,
  not a bare `ARRAY_REF`).
- `BinaryError::BadMagic` docstring: "first four bytes" → "first five".
- `parse_text` doc: state accurately that a `version` header is required
  (vihaco's parser makes headers-before-functions structural, which also
  resolves Copilot's leading-`.version` note).
- Python `InvalidVersionError` mapping: pass `"line N: '<value>'"` (drop the
  duplicated "invalid version" prefix).
- Python `BadMagicError` message: "expected BLQD" → "expected LANES".
- `UnalignedCode`: add a dedicated `UnalignedCodeError(len, width)` Python
  exception instead of reusing the legacy `InvalidCodeSectionLengthError`
  ("multiple of 8").

## Testing

- Rust: round-trip (`parse_text`↔`to_text`, `to_binary`↔`from_binary`),
  `Display`-based emitter, `Resolve` (version header, `@main` body), validation
  + stack-sim unchanged, updated for the `fn @main` text in fixtures.
- CLI/FFI: `cli_integration`, `c_api`, `smoke.c`, `test_smoke.sh` updated to the
  new text format.
- Python: `test_bytecode` (new text format, exception mappings), full pytest
  suite (compiler pipeline) must stay green.
- `cargo fmt` / `clippy -D warnings`, C-header freshness, ruff/black/isort.

## Risks

- **Text format change** is user-visible; every `.sst` fixture/test must move to
  `fn @main { … }`. Mechanical but broad.
- The `LanesHeader`/`Resolve` glue is new surface; keep it minimal and well
  tested.
- `Module`'s many unused fields are inert but must be constructed consistently
  (helper constructor `Program::from_code(version, code)`).
