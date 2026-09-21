//! Vihaco-backed instruction set (ISA) for the Bloqade Lanes bytecode.
//!
//! This is the bytecode instruction set, built on the [`vihaco`] virtual-ISA
//! framework (the migration off the original hand-rolled format), per
//! <https://github.com/QuEraComputing/bloqade-lanes/issues/769>.
//!
//! The instruction set is defined once in [`def`] as a `#[derive(Instruction)]`
//! enum, which generates binary encode/decode
//! ([`vihaco::instruction::WriteBytes`] / [`vihaco::instruction::FromBytes`]) —
//! a 1-byte opcode followed by a little-endian payload, zero-padded to a fixed
//! [`INSTRUCTION_WIDTH`]-byte word, so a program is simply N concatenated words.
//!
//! Text (`.sst`) parsing lives in [`syntax`], which layers vihaco 0.4's pattern
//! parser on top of the same enum.
//!
//! Those words are then framed by vihaco's own section containers — binary
//! `VHBC` and text `sst v1` — carrying a single root section whose header holds
//! the version. vihaco ships readers for both and writers for neither, so
//! [`container`] owns the emitters. This is intentionally *not* compatible with
//! either container Bloqade Lanes used before (`BLQD`, then `LANES`).
//!
//! ## Two dialects, one flat enum
//!
//! In text, every instruction carries a dialect head: [`CPU_HEAD`] (`cpu.pop`,
//! `cpu.halt`, `cpu.const_int 42`) for the stack ops and [`LANES_HEAD`]
//! (`lanes.move 2`) for the device ops. In Rust they are one flat
//! [`Instruction`] enum; [`syntax`] holds the per-dialect mirror enums that
//! carry the text patterns and fold back into it.
//!
//! The stack ops mirror vihaco-cpu's but are declared natively. vihaco-cpu 0.4
//! is a runtime *component*: its instruction enums implement `Parse` and carry
//! runtime values, but neither implements `WriteBytes`/`FromBytes`/`OpCode`, so
//! nesting one in an encodable ISA is no longer possible. The `cpu.` head is
//! kept to signal where the semantics come from.

pub mod container;
pub mod def;
pub mod device;
pub mod machine;
pub mod parse_helpers;
pub mod program;
pub mod syntax;
pub mod text;
pub mod validate;

pub use def::{INSTRUCTION_WIDTH, Instruction};
pub use program::{LanesInfo, Program, from_code};
pub use syntax::{CPU_HEAD, LANES_HEAD};
pub use text::{parse_text, to_text};
