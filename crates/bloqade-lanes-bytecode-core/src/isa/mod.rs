//! Vihaco-backed instruction set (ISA) for the Bloqade Lanes bytecode.
//!
//! This is the bytecode instruction set, built on the [`vihaco`] virtual-ISA
//! framework (the migration off the original hand-rolled format), per
//! <https://github.com/QuEraComputing/bloqade-lanes/issues/769>.
//!
//! The instruction set is defined once as a `#[derive(Instruction, Parse)]`
//! enum; vihaco's derive macros then generate:
//!
//! - binary encode/decode ([`vihaco::instruction::WriteBytes`] /
//!   [`vihaco::instruction::FromBytes`]) — a 1-byte opcode followed by a
//!   little-endian payload, zero-padded to a fixed [`INSTRUCTION_WIDTH`]-byte
//!   word, so a program is simply N concatenated words;
//! - a text (`.sst`) parser ([`vihaco_parser_core::Parse`]).
//!
//! This adopts vihaco's **native** byte layout (the issue #769 decision); it is
//! intentionally *not* compatible with the original `BLQD` container that the
//! hand-rolled bytecode used.
//!
//! ## CPU ops are reused from `vihaco-cpu`
//!
//! Rather than re-defining stack/const opcodes, the [`Instruction::Cpu`]
//! variant nests the entire [`vihaco_cpu::Instruction`] set (a stack machine
//! with `const.<type>`, `dup`, `halt`, arithmetic, …), with parsing
//! `#[delegate]`d to vihaco-cpu's own parser and printing via its `Display`.
//! Three stack ops stay lanes-native because vihaco-cpu can't round-trip them
//! through text:
//!
//! - `pop`, `swap` — vihaco-cpu has no such opcodes;
//! - `return` — vihaco-cpu's `Return` is parser-deferred to an orchestrator
//!   (`ret` does not parse standalone), and lanes needs a terminator that
//!   round-trips.
//!
//! Consequence: CPU text syntax is now vihaco-cpu's (`const.i64 42`,
//! `const.f64 1.5`, `dup`, `halt`), not the legacy `const_int` / `const_float`.

pub mod def;
pub mod parse_helpers;
pub mod program;
pub mod text;
pub mod validate;

pub use def::{INSTRUCTION_WIDTH, Instruction};
pub use program::{LanesInfo, Program, from_code};
pub use text::{parse_text, to_text};
