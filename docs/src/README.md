# Bloqade Lanes

Bloqade Lanes is a component of QuEra's Neutral Atom SDK. It compiles quantum circuits down to physical atom movement instructions for neutral atom quantum processors.

## What's in this book

- **[Architecture Specification](arch/archspec.md)** — the `ArchSpec` JSON format that defines device topology, transport buses, zones, and AOD paths
- **[Bytecode Instruction Set](bytecode/inst-spec.md)** — the fixed-width instruction encoding, opcode layout, and per-instruction reference

## Crate documentation

The Rust API documentation is generated separately via `cargo doc`:

- [`bloqade-lanes-bytecode-core`](https://queracomputing.github.io/bloqade-lanes/api/bloqade_lanes_bytecode_core/) — pure Rust: bytecode format, arch spec, validation
- [`bloqade-lanes-bytecode-cli`](https://queracomputing.github.io/bloqade-lanes/api/bloqade_lanes_bytecode/) — CLI tool and C FFI library

## Repository

Source code: [github.com/QuEraComputing/bloqade-lanes](https://github.com/QuEraComputing/bloqade-lanes)
