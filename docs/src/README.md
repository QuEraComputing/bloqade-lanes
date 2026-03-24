# Bloqade Lanes

Bloqade Lanes is a component of QuEra's Neutral Atom SDK. It compiles quantum circuits down to physical atom movement instructions for neutral atom quantum processors.

## What's in this book

- **[Architecture Specification](arch/archspec.md)** — the `ArchSpec` JSON format that defines device topology, transport buses, zones, and AOD paths
- **[Bytecode Instruction Set](bytecode/inst-spec.md)** — the fixed-width instruction encoding, opcode layout, and per-instruction reference
- **[Instruction Quick Reference](bytecode/inst-quick-ref.md)** — compact summary of all 24 instructions with opcodes and stack effects
- **[CLI Reference](bytecode/cli.md)** — the `bloqade-bytecode` CLI tool and C FFI library

## Crate documentation

The Rust API documentation is generated separately via `cargo doc`:

- [`bloqade-lanes-bytecode-core`](api/bloqade_lanes_bytecode_core/index.html) — pure Rust: bytecode format, arch spec, validation
- [`bloqade-lanes-bytecode-cli`](api/bloqade_lanes_bytecode/index.html) — CLI tool and C FFI library

## Repository

Source code: [github.com/QuEraComputing/bloqade-lanes](https://github.com/QuEraComputing/bloqade-lanes)
