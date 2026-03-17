# CLAUDE.md - Bloqade Lanes

## Project Overview

Bloqade Lanes is a component of QuEra's Neutral Atom SDK. It compiles quantum circuits down to physical atom movement instructions for neutral atom quantum processors (targeting the Atom Computing Gemini architecture). The compilation pipeline is: Circuit → Place (logical placement) → Move (physical moves) → Squin/Stim IR.

This is a hybrid Rust + Python package. The Rust crates provide a bytecode format, architecture specification, CLI tool, and C FFI library. The Python layer provides the quantum circuit compilation pipeline built on the Kirin IR framework.

## Build & Dependencies

- **Package manager**: `uv` (Python) + `cargo` (Rust)
- **Build backend**: Maturin (PyO3 bindings for Rust → Python)
- **Python**: >= 3.10 (tested on 3.10, 3.11, 3.12)
- **Rust**: stable toolchain
- **Source layout**: `python/bloqade/lanes/` (Python), `crates/` (Rust)
- **Key Python deps**: kirin-toolchain (IR framework), bloqade-circuit, bloqade-geometry, rustworkx, numpy, scipy
- **Key Rust deps**: pyo3, serde, clap, cbindgen, thiserror

### Setup

```bash
uv sync --dev --all-extras --index-strategy=unsafe-best-match
# or: just sync
```

For full development (including CLI + C library):
```bash
just develop
```

For Python extension only:
```bash
just develop-python
```

## Common Commands

All tasks use `just` (rust-just):

```bash
# Python
just coverage          # Run Python tests with coverage + generate XML report
just coverage-run      # Run Python tests only
just coverage-html     # Generate HTML coverage report
just demo              # Run all demo scripts
just doc               # Build and open documentation site in browser
just doc-all           # Build documentation site (book + Rust API)
just doc-book          # Build mdBook only
just doc-rust          # Build Rust API docs only
just test-python       # Run Python tests via pytest

# Rust
just test-rust         # Run Rust tests (core + cli crates)
just check             # Type-check Rust (no linking)
just format            # Format Rust code
just format-check      # Check Rust formatting
just lint              # Run clippy on core + cli crates
just check-header      # Verify C header freshness
just test-c-ffi        # Build and run C FFI smoke test
just cli-smoke-test    # CLI bytecode validation tests

# Combined
just test              # Run all tests (Rust + Python)

# Packaging
just build-cli         # Build CLI in release mode
just stage-clib        # Stage CLI + C lib for wheel packaging
just build-wheel       # Build Python wheel with bundled CLI + C lib
just develop           # Dev install with bundled CLI + C lib
```

Direct test run: `uv run coverage run -m pytest python/tests`

## Linting & Formatting

Pre-commit hooks enforce all checks. The CI lint pipeline runs:

### Python
- **isort** (profile=black, src_paths=python/bloqade)
- **black** (line-length=88)
- **ruff** (target=py312)
- **pyright** (on python/)

### Rust
- **cargo fmt** (all crates)
- **cargo clippy** (core + cli crates, -D warnings)

Run manually:
```bash
# Python
uv run black python
uv run isort python
uv run ruff check python
uv run pyright python

# Rust
cargo fmt --all
cargo clippy -p bloqade-lanes-bytecode-core -p bloqade-lanes-bytecode-cli --all-targets -- -D warnings
```

## Code Conventions

- Absolute imports from `bloqade.lanes` namespace
- snake_case for files/functions, PascalCase for classes
- Extensive type annotations (enforced by pyright)
- Heavy use of Python dataclasses
- Built on Kirin IR framework: dialects, analysis passes, rewrite passes
- Rust code follows standard Rust conventions (enforced by clippy + rustfmt)

## Project Structure

```
bloqade-lanes/
├── Cargo.toml              # Rust workspace root
├── Cargo.lock
├── pyproject.toml           # Maturin build config + Python project metadata
├── justfile                 # Task automation
├── crates/                  # Rust workspace
│   ├── bloqade-lanes-bytecode-core/     # Pure Rust: bytecode format, arch spec, validation
│   ├── bloqade-lanes-bytecode-python/   # PyO3 bindings (cdylib → _native module)
│   └── bloqade-lanes-bytecode-cli/      # CLI tool + C FFI library
├── python/bloqade/lanes/
│   ├── __init__.py          # Main package exports
│   ├── device.py            # Device interface
│   ├── logical_mvp.py       # Entry point for logical compilation
│   ├── types.py             # Custom IR types
│   ├── arch/                # Architecture definitions (Gemini)
│   ├── analysis/            # Analysis passes (atom state, placement, layout)
│   ├── dialects/            # Kirin IR dialects (move, place)
│   ├── heuristics/          # Layout/scheduling heuristics
│   ├── layout/              # Layout representation (ArchSpec, Word, encoding, PathFinder)
│   ├── rewrite/             # Compilation passes (circuit→place→move→squin)
│   ├── validation/          # Validation passes
│   ├── visualize/           # Visualization and debugging tools
│   └── bytecode/            # Rust-backed bytecode module (PyO3 bindings)
│       ├── __init__.py      # Re-exports from _native
│       ├── _native.pyi      # Type stubs for Rust extension
│       ├── arch.py          # Python arch spec utilities
│       ├── exceptions.py    # Exception types
│       └── _clib_path.py    # C library path helpers
├── python/tests/            # Python tests
│   ├── bytecode/            # Bytecode-specific tests
│   └── ...                  # Tests mirror python/bloqade/lanes structure
├── tests/                   # Rust integration tests
├── examples/                # Architecture specs and sample bytecode programs
├── scripts/                 # Build/test utility scripts
├── demo/                    # Python demo scripts
└── dist-data/               # Staged artifacts for wheel packaging (gitignored)
```

## Commit Message Convention

This project follows [Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Purpose |
|------|---------|
| `feat` | New functionality (SemVer MINOR) |
| `fix` | Bug fix (SemVer PATCH) |
| `docs` | Documentation only |
| `style` | Formatting, whitespace (no logic change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `build` | Build system or external dependencies |
| `ci` | CI configuration and scripts |
| `chore` | Other changes that don't modify src or test files |

### Scopes

Use the crate or subsystem name as scope when the change is focused:

- `core`, `python`, `cli` — crate-level scopes
- `arch`, `bytecode`, `ffi` — subsystem scopes
- Omit scope for cross-cutting changes

Examples:
```
feat(bytecode): add new SWAP instruction encoding
fix(python): correct exception mapping for LaneGroupError
docs: update AGENT.md with commit conventions
refactor(arch): simplify Grid validation logic
ci: add pyright to lint workflow
```

### Breaking Changes

Signal breaking changes in one of two ways:

1. **Footer**: Add `BREAKING CHANGE: <description>` in the commit footer
2. **Type suffix**: Append `!` before the colon — e.g., `feat(bytecode)!: redesign instruction encoding`

Breaking changes map to SemVer MAJOR.

### Rules

- The description MUST immediately follow the type/scope prefix colon and space
- The body, if present, MUST begin one blank line after the description
- Footers, if present, MUST begin one blank line after the body
- Use imperative mood in the description ("add", not "added" or "adds")

### Pull Request Labels

- Tag PRs with the `breaking` label when they contain breaking changes
