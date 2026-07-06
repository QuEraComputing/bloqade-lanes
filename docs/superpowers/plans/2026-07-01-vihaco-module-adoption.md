# vihaco Module Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-point the bytecode ISA onto vihaco's `Module` + text parser, with a hand-rolled `LANES` binary codec and a round-trippable `fn @main` text emitter, while addressing the PR #770 review.

**Architecture:** The program object becomes `Module<Instruction, Value, Type, LanesInfo>` (single `@main` function, version in `extra`). Text reading uses vihaco's `ParsedModule` parser + a `Resolve` impl; text writing uses a small emitter of the same `fn @main { … }` grammar routed through a `Display` impl on `Instruction`; binary stays our `LANES` codec over `Module`. The module is renamed `vihaco_isa` → `isa`.

**Tech Stack:** Rust (vihaco, vihaco-cpu, vihaco-parser, chumsky), PyO3, Python (kirin), C FFI (cbindgen).

## Global Constraints

- All Rust edits keep `cargo fmt --all -- --check` and `cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli -p bloqade-lanes-bytecode-python --all-targets -- -D warnings` clean.
- The Python-facing factory API of `PyInstruction`/`PyProgram` (method names + signatures) MUST NOT change — the compiler's `encode.py` and `_native.pyi` depend on it.
- Binary word width stays `INSTRUCTION_WIDTH = 17`; magic stays `b"LANES"`.
- Sandbox: build/test commands that fetch crates or run maturin need the sandbox disabled.
- After any FFI change, the committed C header `crates/bloqade-lanes-bytecode-cli/bloqade_lanes_bytecode.h` must match cbindgen output (`git diff --exit-code` on it).
- Full Python suite (`uv run pytest python/tests/`) must stay green (currently 1433 passed / 9 skipped).

---

### Task 1: Rename `vihaco_isa` module → `isa`

**Files:**
- Rename dir: `crates/bloqade-lanes-bytecode-core/src/vihaco_isa/` → `crates/bloqade-lanes-bytecode-core/src/isa/`
- Modify: `crates/bloqade-lanes-bytecode-core/src/lib.rs` (`pub mod vihaco_isa;` → `pub mod isa;`, doc links)
- Modify (path refs): `crates/bloqade-lanes-bytecode-cli/src/main.rs`, `src/ffi/{handles,program,validate}.rs`, `crates/bloqade-lanes-bytecode-python/src/{program_python,instruction_python,errors}.rs`

**Interfaces:**
- Produces: the module is now reachable as `bloqade_lanes_bytecode_core::isa::{Instruction, Program, program, validate}`.

- [ ] **Step 1: Rename the directory and module declaration**

```bash
cd crates/bloqade-lanes-bytecode-core/src
git mv vihaco_isa isa
```
Edit `lib.rs`: `pub mod vihaco_isa;` → `pub mod isa;` and update the two doc-comment references (`[vihaco_isa]` → `[isa]`).

- [ ] **Step 2: Update all references workspace-wide**

Run:
```bash
grep -rl "vihaco_isa" crates/*/src
```
In each hit, replace `vihaco_isa` → `isa` (module paths like `bloqade_lanes_bytecode_core::vihaco_isa::…`, and any `crate::vihaco_isa::…` inside core). Do NOT touch the crate name `bloqade-lanes-bytecode-core` or the `vihaco`/`vihaco_cpu` crate names.

- [ ] **Step 3: Build + test**

Run: `cargo test -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli 2>&1 | grep -E "test result|error"`
Expected: all `test result: ok`, no `error`.

- [ ] **Step 4: fmt + clippy**

Run: `cargo fmt --all && cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli --all-targets -- -D warnings 2>&1 | tail -1`
Expected: `Finished`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(bytecode): rename vihaco_isa module to isa (#769)"
```

---

### Task 2: Add `Display` for `Instruction`; route text printing through it

Single source of truth for the per-instruction mnemonic (closes #544 core side). Text *format* stays flat here; only the source of the mnemonic strings changes.

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/isa/mod.rs` (add `impl Display for Instruction`)
- Modify: `crates/bloqade-lanes-bytecode-core/src/isa/program.rs` (`instruction_to_text` → use `Display`)

**Interfaces:**
- Produces: `impl std::fmt::Display for Instruction` — renders one canonical text line (mnemonic + operands), e.g. `const_loc 0x0000000000000000`, `move 1`, `const.f64 1.5`.

- [ ] **Step 1: Write the failing test** (in `isa/mod.rs` `#[cfg(test)] mod tests`)

```rust
#[test]
fn display_matches_parser_tokens() {
    use std::string::ToString;
    // Every non-CPU variant's Display must re-parse to itself.
    let samples = [
        Instruction::ConstLoc(0x0100_0000),
        Instruction::Move(2),
        Instruction::LocalRz(1),
        Instruction::GlobalR,
        Instruction::NewArray(1, 3, 0),
        Instruction::Pop,
        Instruction::Return,
        Instruction::Cpu(Cpu::Const(Value::F64(1.5))),
        Instruction::Cpu(Cpu::Halt),
    ];
    for inst in samples {
        let text = inst.to_string();
        assert_eq!(parse(&text), inst, "Display/parse mismatch for {text:?}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p bloqade-lanes-bytecode-core isa::tests::display_matches_parser_tokens 2>&1 | tail -5`
Expected: FAIL — `Instruction` doesn't implement `Display`.

- [ ] **Step 3: Implement `Display`** (add to `isa/mod.rs`, moving the body of `program.rs::instruction_to_text` here)

```rust
impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Pop => f.write_str("pop"),
            Instruction::Swap => f.write_str("swap"),
            Instruction::Return => f.write_str("return"),
            Instruction::ConstLoc(v) => write!(f, "const_loc 0x{v:016x}"),
            Instruction::ConstLane(v) => write!(f, "const_lane 0x{v:016x}"),
            Instruction::ConstZone(v) => write!(f, "const_zone 0x{v:08x}"),
            Instruction::InitialFill(a) => write!(f, "initial_fill {a}"),
            Instruction::Fill(a) => write!(f, "fill {a}"),
            Instruction::Move(a) => write!(f, "move {a}"),
            Instruction::LocalRz(a) => write!(f, "local_rz {a}"),
            Instruction::LocalR(a) => write!(f, "local_r {a}"),
            Instruction::GlobalRz => f.write_str("global_rz"),
            Instruction::GlobalR => f.write_str("global_r"),
            Instruction::Cz => f.write_str("cz"),
            Instruction::Measure(a) => write!(f, "measure {a}"),
            Instruction::AwaitMeasure => f.write_str("await_measure"),
            Instruction::NewArray(t, d0, d1) => write!(f, "new_array {t} {d0} {d1}"),
            Instruction::GetItem(n) => write!(f, "get_item {n}"),
            Instruction::SetDetector => f.write_str("set_detector"),
            Instruction::SetObservable => f.write_str("set_observable"),
            // vihaco-cpu owns its own Display (const.f64 / dup / halt / …).
            Instruction::Cpu(cpu) => write!(f, "{cpu}"),
        }
    }
}
```
Add `use std::fmt::Write as _;` is NOT needed (using `write!`/`f.write_str`).

- [ ] **Step 4: Route `program.rs::instruction_to_text` through `Display`**

In `isa/program.rs`, replace the `instruction_to_text` function body with:
```rust
fn instruction_to_text(inst: &Instruction) -> String {
    inst.to_string()
}
```
(Leave `to_text` calling `instruction_to_text` for now; format unchanged.)

- [ ] **Step 5: Run tests**

Run: `cargo test -p bloqade-lanes-bytecode-core isa:: 2>&1 | grep "test result"`
Expected: PASS (new test + all existing round-trip tests still green).

- [ ] **Step 6: fmt + commit**

```bash
cargo fmt --all
git add -A
git commit -m "refactor(bytecode): single-source instruction mnemonics via Display (#544)"
```

---

### Task 3: Adopt vihaco `Module` as `Program`

Container becomes a `Module`; binary codec + validation + PyProgram accessors updated. Text still uses the OLD flat `.version` format (migrated in Task 4). Includes the Copilot `BadMagic` docstring fix.

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/isa/program.rs` (define `LanesInfo`, `Program` alias, `from_code`, rewrite `to_binary`/`from_binary`, keep `parse_text`/`to_text` building/reading the alias)
- Modify: `crates/bloqade-lanes-bytecode-core/src/isa/validate.rs` (iterate `program.code`, read `program.extra.version` if needed)
- Modify: `crates/bloqade-lanes-bytecode-python/src/program_python.rs` (constructor + `version`/`instructions` getters use `Module` fields)
- Modify: `crates/bloqade-lanes-bytecode-cli/src/main.rs` and `src/ffi/program.rs` (call sites: methods → free fns)

**Interfaces:**
- Produces:
  - `pub struct LanesInfo { pub version: Version }` (Debug, Clone, PartialEq, Default, Display)
  - `pub type Program = vihaco::module::Module<Instruction, vihaco::value::Value, vihaco::value::Type, LanesInfo>;`
  - `pub fn from_code(version: Version, code: Vec<Instruction>) -> Program`
  - `to_binary(&Program) -> Vec<u8>`, `from_binary(&[u8]) -> Result<Program, BinaryError>` (free fns, since `Program` is now a type alias — no inherent impls on `Module`)
  - `parse_text(&str) -> Result<Program, TextError>`, `to_text(&Program) -> String` (free fns)
- Consumes: `Instruction: Display` (Task 2).

- [ ] **Step 1: Define `LanesInfo`, the `Program` alias, and `from_code`** (top of `isa/program.rs`)

```rust
use vihaco::module::Module;
use vihaco::value::{Type, Value};

/// Consumer metadata carried in `Module::extra` (vihaco has no version field).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LanesInfo {
    pub version: Version,
}

impl std::fmt::Display for LanesInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "version {}", self.version)
    }
}

/// A Bloqade Lanes program: a vihaco `Module` specialised to our ISA. A single
/// `@main` function's worth of flat code plus the version in `extra`.
pub type Program = Module<Instruction, Value, Type, LanesInfo>;

/// Build a `Program` from a version + flat instruction list. This is the ONE
/// constructor used by both binary and text loading, so all `Program`s built
/// from the same (version, code) compare equal regardless of source.
pub fn from_code(version: Version, code: Vec<Instruction>) -> Program {
    let mut m = Program::default();
    m.code = code;
    m.extra = LanesInfo { version };
    m
}
```

- [ ] **Step 2: Convert `to_binary`/`from_binary`/`parse_text`/`to_text` from methods to free functions over the alias**

The current `impl Program { … }` block no longer works (`Program` is a type alias to a foreign type — no inherent impls). Move those four methods to free `pub fn`s in the module, reading `program.code` / `program.extra.version`. Example for `to_binary`:
```rust
pub fn to_binary(program: &Program) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_LEN + program.code.len() * INSTRUCTION_WIDTH as usize);
    buf.extend_from_slice(MAGIC);
    let packed: u32 = program.extra.version.into();
    buf.extend_from_slice(&packed.to_le_bytes());
    for inst in &program.code {
        inst.write_bytes(&mut buf).expect("writing to Vec is infallible");
    }
    buf
}
```
`from_binary`: decode as today, then `Ok(from_code(version, instructions))`. `parse_text`/`to_text`: keep the existing flat logic but read `program.code`/`program.extra.version` and build via `from_code`.

- [ ] **Step 3: Fix the `BadMagic` docstring (Copilot #1)**

In `isa/program.rs`, the `BinaryError::BadMagic` doc comment "First four bytes were not [`MAGIC`]." → "First five bytes were not [`MAGIC`]."

- [ ] **Step 4: Update `validate.rs` to iterate `program.code`**

In `isa/validate.rs`, every `program.instructions` → `program.code`. The `validate`/`validate_structure`/`simulate_stack` signatures keep `&Program`. `Program::validate`/`validate_all` were inherent methods on the old struct — convert to free fns `validate_all(program, arch)` OR keep as methods via an extension trait. Simplest: make them free fns `pub fn validate_all(program: &Program, arch: Option<&ArchSpec>) -> Vec<ValidationError>` and update call sites (CLI, python) accordingly.

- [ ] **Step 5: Update test constructors in core**

In `isa/program.rs` and `isa/validate.rs` tests, replace every `Program { version: …, instructions: vec![…] }` with `from_code(Version::new(..), vec![…])`, and `p.to_binary()` → `to_binary(&p)`, `Program::from_binary(b)` → `from_binary(b)`, `p.validate(a)` → `validate_all(&p, a)` (or the chosen name), `.instructions` → `.code`.

- [ ] **Step 6: Update `program_python.rs`**

- `PyProgram::new`: build via `rs_prog::from_code(Version::new(version.0, version.1), code)`.
- `from_text`/`to_text`/`from_binary`/`to_binary`: call the free fns (`rs_prog::to_binary(&self.inner)` etc.).
- `version` getter: `self.inner.extra.version.{major,minor}`.
- `instructions` getter: iterate `self.inner.code`.
- `validate`: call `rs_val::validate_structure(&self.inner)` + `rs_val::validate(&self.inner, arch_ref)` + `rs_val::simulate_stack(...)` (free fns; unchanged names).

- [ ] **Step 6b: Update CLI + FFI call sites (methods → free fns)**

In `crates/bloqade-lanes-bytecode-cli/src/main.rs` and `src/ffi/program.rs`:
- `program.to_binary()` → `to_binary(&program)`; `Program::from_binary(b)` → `from_binary(b)`
- `program.to_text()` → `to_text(&program)`; `Program::parse_text(s)` → `parse_text(s)`
- `program.instructions` / `.instructions.len()` → `program.code` / `.code.len()`
- `program.version.{major,minor}` → `program.extra.version.{major,minor}`
- CLI `validate` command: `program.validate(...)` (if used) → the free `validate_all`/`validate` fns.

Imports: bring the free fns into scope (`use bloqade_lanes_bytecode_core::isa::{Program, program::{to_binary, from_binary}, text::{parse_text, to_text}};` — adjust to final module paths after Task 4; in Task 3 they still live in `program`).

- [ ] **Step 7: Build + test core, then CLI + python bindings compile**

Run: `cargo test -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli 2>&1 | grep "test result"` → all ok.
Run: `cargo check -p bloqade-lanes-bytecode-python 2>&1 | tail -1` → `Finished`.

- [ ] **Step 8: Rebuild extension + run bytecode pytest**

Run (sandbox off): `uv run maturin develop 2>&1 | tail -1 && uv run pytest python/tests/bytecode/ -q 2>&1 | tail -3`
Expected: `215 passed`.

- [ ] **Step 9: fmt + clippy + commit**

```bash
cargo fmt --all
cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli -p bloqade-lanes-bytecode-python --all-targets -- -D warnings 2>&1 | tail -1
git add -A
git commit -m "refactor(bytecode): use vihaco Module as the Program object (#769)"
```

---

### Task 4: Use vihaco's text parser + `fn @main` emitter; migrate text fixtures

Replaces the hand-rolled `.version`+flat text with vihaco's `ParsedModule` grammar. **Visible format change.**

**Files:**
- Create: `crates/bloqade-lanes-bytecode-core/src/isa/text.rs` (`LanesHeader` + parser, `Resolve` impl, `parse_text`, `to_text`)
- Modify: `crates/bloqade-lanes-bytecode-core/src/isa/mod.rs` (`pub mod text;`)
- Modify: `crates/bloqade-lanes-bytecode-core/src/isa/program.rs` (remove the old flat `parse_text`/`to_text`; keep binary)
- Modify (fixtures → new format): `crates/bloqade-lanes-bytecode-cli/tests/cli_integration.rs`, `crates/bloqade-lanes-bytecode-cli/tests/c_api.rs`, `tests/c_smoke/smoke.c`, `scripts/test_smoke.sh`, `python/tests/bytecode/test_bytecode.py`, and any `.sst` literals in core tests.

**Interfaces:**
- Produces: `parse_text(&str) -> Result<Program, TextError>` and `to_text(&Program) -> String` over the grammar:
  ```
  version <major>.<minor>;
  fn @main() {
    <instruction>
    ...
  }
  ```

- [ ] **Step 1: Write the failing round-trip test** (`isa/text.rs` tests)

```rust
#[test]
fn text_round_trips_fn_main() {
    let src = "version 1.2;\nfn @main() {\n  const_loc 0x0000000000000000\n  initial_fill 1\n  halt\n}\n";
    let p = parse_text(src).unwrap();
    assert_eq!(p.extra.version, Version::new(1, 2));
    assert_eq!(p.code.len(), 3);
    assert_eq!(parse_text(&to_text(&p)).unwrap(), p);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p bloqade-lanes-bytecode-core isa::text 2>&1 | tail -5`
Expected: FAIL (module `text` / `parse_text` not found there).

- [ ] **Step 3: Implement `LanesHeader` + parser**

```rust
use chumsky::prelude::*;
use vihaco::syntax::ParsedModule;
use vihaco::syntax::Resolve;
use vihaco_parser_core::Parse;

/// The only header we support: `version <major>.<minor>`.
#[derive(Debug, Clone, PartialEq)]
pub enum LanesHeader {
    Version(Version),
}

impl<'src> Parse<'src> for LanesHeader {
    fn parser() -> impl chumsky::Parser<'src, &'src str, Self, chumsky::extra::Err<chumsky::error::Simple<'src, char>>> {
        let uint = any().filter(|c: &char| c.is_ascii_digit()).repeated().at_least(1).collect::<String>();
        just("version")
            .ignore_then(chumsky::text::whitespace())
            .ignore_then(uint.clone().then_ignore(just('.')).then(uint))
            .try_map(|(maj, min), span| {
                let major = maj.parse::<u16>().map_err(|_| Simple::new(None, span))?;
                let minor = min.parse::<u16>().map_err(|_| Simple::new(None, span))?;
                Ok(LanesHeader::Version(Version::new(major, minor)))
            })
    }
}
```

- [ ] **Step 4: Implement `Resolve` + `parse_text`**

```rust
struct LanesResolver;

impl Resolve<Instruction, LanesHeader> for LanesResolver {
    type Module = Program;
    fn resolve_module(&mut self, parsed: ParsedModule<Instruction, LanesHeader>) -> eyre::Result<Program> {
        let version = parsed
            .headers
            .iter()
            .find_map(|h| match h { LanesHeader::Version(v) => Some(*v) })
            .ok_or_else(|| eyre::eyre!("missing version header"))?;
        let func = match parsed.functions.as_slice() {
            [f] => f,
            _ => eyre::bail!("expected exactly one function (@main)"),
        };
        // default resolve_body: all lanes instructions are Direct; Raw is an error.
        let code = self.resolve_body(func.body.clone())?;
        Ok(from_code(version, code))
    }
}

pub fn parse_text(src: &str) -> Result<Program, TextError> {
    let parsed = ParsedModule::<Instruction, LanesHeader>::parser()
        .parse(src)
        .into_result()
        .map_err(|_| TextError::BadInstruction { line: 0, text: "parse error".into() })?;
    LanesResolver
        .resolve_module(parsed)
        .map_err(|e| map_resolve_err(&e))
}
```
Add `map_resolve_err` translating the eyre messages to `TextError::{MissingVersion, BadInstruction}` (match on the message text produced above). Keep `TextError` variants as-is.

- [ ] **Step 5: Implement `to_text`**

```rust
pub fn to_text(program: &Program) -> String {
    let mut out = format!("version {}.{};\nfn @main() {{\n", program.extra.version.major, program.extra.version.minor);
    for inst in &program.code {
        out.push_str("  ");
        out.push_str(&inst.to_string()); // Display from Task 2
        out.push('\n');
    }
    out.push_str("}\n");
    out
}
```

- [ ] **Step 6: Wire module + remove old flat text**

In `isa/mod.rs` add `pub mod text;`. Remove the old `parse_text`/`to_text` from `program.rs`. Re-export from `text`: in `isa/mod.rs`, `pub use text::{parse_text, to_text};` (and keep `pub use program::Program;`). Update `program_python.rs`, `main.rs`, `ffi/program.rs` calls to the new paths if needed (names unchanged).

- [ ] **Step 7: Run core text tests**

Run: `cargo test -p bloqade-lanes-bytecode-core isa:: 2>&1 | grep "test result"`
Expected: PASS.

- [ ] **Step 8: Migrate CLI/FFI/Python text fixtures to the new grammar**

Every embedded program literal moves from:
```
.version 1.0\nconst_loc 0x..\n...\nhalt\n
```
to:
```
version 1.0;\nfn @main() {\n  const_loc 0x..\n  ...\n  halt\n}\n
```
Files: `cli_integration.rs`, `c_api.rs`, `smoke.c`, `test_smoke.sh` (the heredocs), `test_bytecode.py`. Assertions on disassembled output (`.stdout(contains("const.i64 42"))`) still hold; add/adjust any that asserted the leading `.version` line to `version 1.0;` and the `fn @main()` wrapper where a full-text compare is done.

- [ ] **Step 9: Verify Rust CLI/FFI + C smoke + python**

Run (sandbox off):
```bash
cargo test -p bloqade-lanes-bytecode-cli 2>&1 | grep "test result"
./scripts/test_smoke.sh 2>&1 | tail -1
cargo build -p bloqade-lanes-bytecode-cli && cc -I crates/bloqade-lanes-bytecode-cli tests/c_smoke/smoke.c -L target/debug -lbloqade_lanes_bytecode -o /tmp/c_smoke && DYLD_LIBRARY_PATH=target/debug /tmp/c_smoke | tail -1
uv run maturin develop 2>&1 | tail -1 && uv run pytest python/tests/bytecode/ -q 2>&1 | tail -1
```
Expected: all green; smoke `passed, 0 failed`; C smoke `tests passed`; pytest `215 passed`.

- [ ] **Step 10: fmt + clippy + header + commit**

```bash
cargo fmt --all
cargo build -p bloqade-lanes-bytecode-cli && git diff --exit-code crates/bloqade-lanes-bytecode-cli/bloqade_lanes_bytecode.h && echo HEADER OK
cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli -p bloqade-lanes-bytecode-python --all-targets -- -D warnings 2>&1 | tail -1
git add -A
git commit -m "feat(bytecode): use vihaco text parser + fn @main emitter (#769, #544)"
```

---

### Task 5: Add the measurement-result value tag (#547)

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-core/src/isa/validate.rs` (`tag` module + `await_measure` push)

**Interfaces:**
- Produces: `tag::MEASUREMENT_RESULT: u8 = 0x9`.

- [ ] **Step 1: Write the failing test** (`isa/validate.rs` tests)

```rust
#[test]
fn await_measure_pushes_measurement_result() {
    // const_zone, measure 1, await_measure — the awaited value carries the
    // measurement-result tag, so a following set_detector (wants ARRAY_REF)
    // now type-mismatches on MEASUREMENT_RESULT rather than silently matching.
    let p = from_code(Version::new(1, 0), vec![
        Instruction::ConstZone(0),
        Instruction::Measure(1),
        Instruction::AwaitMeasure,
    ]);
    // await_measure must push the measurement-result tag.
    assert_eq!(tag::MEASUREMENT_RESULT, 0x9);
    // Sanity: simulate cleanly (no underflow/mismatch) for the measure→await chain.
    assert!(simulate_stack(&p, None).iter().all(|e|
        !matches!(e, ValidationError::StackUnderflow { .. } | ValidationError::TypeMismatch { .. })));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p bloqade-lanes-bytecode-core isa::validate::tests::await_measure_pushes_measurement_result 2>&1 | tail -5`
Expected: FAIL (`tag::MEASUREMENT_RESULT` not found).

- [ ] **Step 3: Add the tag + use it**

In `isa/validate.rs` `mod tag`, add `pub const MEASUREMENT_RESULT: u8 = 0x9;`. In `dispatch`, change `AwaitMeasure` to push `tag::MEASUREMENT_RESULT` instead of `tag::ARRAY_REF`:
```rust
Instruction::AwaitMeasure => {
    self.pop_typed(tag::MEASURE_FUTURE);
    self.push(tag::MEASUREMENT_RESULT, None);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p bloqade-lanes-bytecode-core isa::validate 2>&1 | grep "test result"`
Expected: PASS. (If a pre-existing test asserted `await_measure` yields an array usable by `set_detector`, update it — `set_detector` still wants `ARRAY_REF`; a measurement result is no longer that. This is the intended semantic per #547/#776.)

- [ ] **Step 5: fmt + commit**

```bash
cargo fmt --all
git add -A
git commit -m "feat(bytecode): add MEASUREMENT_RESULT value tag to stack simulator (#547)"
```

---

### Task 6: Python diagnostic fixes (Copilot #3, #4, #5)

**Files:**
- Modify: `crates/bloqade-lanes-bytecode-python/src/errors.rs` (InvalidVersion mapping; UnalignedCode mapping)
- Modify: `python/bloqade/lanes/bytecode/exceptions.py` (`BadMagicError` message; add `UnalignedCodeError`)

**Interfaces:**
- Produces: `UnalignedCodeError(length: int, width: int)` Python exception (subclass of `ProgramError`).

- [ ] **Step 1: Fix `BadMagicError` message + add `UnalignedCodeError`** (`exceptions.py`)

`BadMagicError.__init__`: change the message from `expected BLQD` to `expected LANES`. Add:
```python
class UnalignedCodeError(ProgramError):
    """Binary code region length is not a whole number of instruction words."""
    def __init__(self, length: int, width: int):
        self.length = length
        self.width = width
        super().__init__(f"code length {length} is not a multiple of {width}")
```

- [ ] **Step 2: Fix the Rust→Python mappings** (`errors.rs`)

- `TextError::InvalidVersion { line, value }` → `InvalidVersionError(format!("line {line}: '{value}'"))` (drop the duplicated "invalid version" wording).
- `BinaryError::UnalignedCode { len }` → `UnalignedCodeError(len, INSTRUCTION_WIDTH)`; import `use bloqade_lanes_bytecode_core::isa::INSTRUCTION_WIDTH;` and pass `(len, INSTRUCTION_WIDTH as usize)`.

- [ ] **Step 3: Add/adjust Python tests** (`test_bytecode.py`)

```python
def test_bad_magic_message_mentions_lanes(self):
    with pytest.raises(BadMagicError) as e:
        Program.from_binary(b"XXXXX\x00\x00\x00\x00")
    assert "LANES" in str(e.value)
```
(Import `UnalignedCodeError` where the binary-error tests live; add one asserting an unaligned buffer raises it.)

- [ ] **Step 4: Rebuild + test**

Run (sandbox off): `uv run maturin develop 2>&1 | tail -1 && uv run pytest python/tests/bytecode/ -q 2>&1 | tail -1`
Expected: all passed.

- [ ] **Step 5: lint + commit**

```bash
uv run ruff check python && uv run black --check python && uv run isort --check python
git add -A
git commit -m "fix(python): correct BadMagic/InvalidVersion/UnalignedCode diagnostics (#770 review)"
```

---

### Task 7: Full verification sweep

**Files:** none (verification only).

- [ ] **Step 1: Rust workspace**

Run: `cargo test --workspace 2>&1 | grep -E "test result: FAILED|error\[" ; echo done`
Expected: only `done` (no failures).

- [ ] **Step 2: fmt + clippy + header**

Run:
```bash
cargo fmt --all -- --check && echo FMT_OK
cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli -p bloqade-lanes-bytecode-python --all-targets -- -D warnings 2>&1 | tail -1
cargo build -p bloqade-lanes-bytecode-cli && git diff --exit-code crates/bloqade-lanes-bytecode-cli/bloqade_lanes_bytecode.h && echo HEADER_OK
```

- [ ] **Step 3: Smoke + C-FFI**

Run: `./scripts/test_smoke.sh 2>&1 | tail -1` and the C smoke compile/run from Task 4 Step 9.

- [ ] **Step 4: Full Python suite**

Run (sandbox off): `uv run maturin develop 2>&1 | tail -1 && uv run pytest python/tests/ -q 2>&1 | tail -2`
Expected: `1433 passed, 9 skipped` (or higher — no regressions).

- [ ] **Step 5: Push**

```bash
git push
```
Then reply on PR #770 review threads pointing to the commits that address each.
