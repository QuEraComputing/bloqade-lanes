# CLAUDE.md - Bloqade Lanes

## Project Overview

Bloqade Lanes is a component of QuEra's Neutral Atom SDK. It compiles quantum circuits down to physical atom movement instructions for neutral atom quantum processors (targeting the Atom Computing Gemini architecture). The compilation pipeline is: Circuit → Place (logical placement) → Move (physical moves) → Squin/Stim IR.

## Build & Dependencies

- **Package manager**: `uv`
- **Build backend**: Hatchling
- **Python**: >= 3.10 (tested on 3.10, 3.11, 3.12)
- **Source layout**: `src/bloqade/lanes/`
- **Key deps**: kirin-toolchain (IR framework), bloqade-circuit, bloqade-geometry, rustworkx, numpy

### Setup

```bash
uv sync --dev --all-extras --index-strategy=unsafe-best-match
# or: just sync
```

## Common Commands

All tasks use `just` (rust-just):

```bash
just coverage          # Run tests with coverage + generate XML report
just coverage-run      # Run tests only: coverage run -m pytest test
just coverage-html     # Generate HTML coverage report
just demo              # Run all demo scripts
just doc               # Serve mkdocs locally
```

Direct test run: `uv run coverage run -m pytest test`

## Linting & Formatting

Pre-commit hooks enforce all checks. The CI lint pipeline runs:

- **isort** (profile=black, src_paths=src/bloqade)
- **black** (line-length=88)
- **ruff** (target=py312)
- **pyright** (on src/ and test/)

Run manually:
```bash
uv run black src test
uv run isort src
uv run ruff check src
uv run pyright src test
```

## Code Conventions

- Absolute imports from `bloqade.lanes` namespace
- snake_case for files/functions, PascalCase for classes
- Extensive type annotations (enforced by pyright)
- Heavy use of Python dataclasses
- Built on Kirin IR framework: dialects, analysis passes, rewrite passes

## Project Structure

```
src/bloqade/lanes/
├── arch/           # Architecture definitions (Gemini)
├── analysis/       # Analysis passes (atom state, placement, layout)
├── dialects/       # Kirin IR dialects (move, place)
├── heuristics/     # Layout/scheduling heuristics
├── layout/         # Layout representation (ArchSpec, Word, encoding, PathFinder)
├── rewrite/        # Compilation passes (circuit→place→move→squin)
├── validation/     # Validation passes
├── visualize/      # Visualization and debugging tools
├── device.py       # Device interface
├── logical_mvp.py  # Entry point for logical compilation
└── types.py        # Custom IR types
test/               # Tests mirror src structure
demo/               # Example/demo scripts
```
