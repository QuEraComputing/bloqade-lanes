//! Vihaco-backed instruction set (ISA) for the Bloqade Lanes bytecode.
//!
//! Built on the [`vihaco`] virtual-ISA framework, per
//! <https://github.com/QuEraComputing/bloqade-lanes/issues/769>.
//!
//! ## Two devices, one machine
//!
//! A lanes program runs on a composite of two vihaco devices, which is how
//! vihaco intends a VM to be assembled (PPVM is built the same way):
//!
//! | Device | Supplies |
//! |---|---|
//! | `cpu` | vihaco-cpu's `CPU` component — stack, constants, arithmetic, control flow, the heap allocator |
//! | `lanes` | atom movement, gates, measurement, arrays, and the `pop`/`swap` the CPU lacks |
//!
//! [`device`] declares the lanes device with [`vihaco::component!`];
//! [`machine`] composes the pair and routes operands between the CPU stack and
//! the device. Every instruction is spelled `<device>::<dialect>.<mnemonic>` —
//! `lanes::lanes.move 2`, `cpu::cpu.halt`.
//!
//! ## Why the binary codec is a separate enum
//!
//! vihaco-cpu 0.4 is a runtime *component*, not an opcode library: its
//! instruction enums implement `Parse` and carry runtime values, but none
//! implements `WriteBytes`/`FromBytes`/`OpCode` — and the enum `#[composite]`
//! generates derives only `Debug`/`Clone`. So neither half of the machine can
//! be encoded directly. [`bytecode`] declares a parallel instruction set that
//! *does* derive [`vihaco::Instruction`], with explicit conversions either way
//! and a round-trip test to keep it honest. PPVM carries the same mirror, for
//! the same reason.
//!
//! An encoded word is a device byte, an instruction byte, and up to 12 bytes
//! of operand, zero-padded to a fixed width ([`instruction_width`]) — so a
//! program is N concatenated words.
//!
//! ## Containers
//!
//! Those words are framed by vihaco's own section containers — binary `VHBC`
//! and text `sst v1` — carrying one root section whose header holds the
//! version. vihaco ships readers for both and writers for neither, so
//! [`container`] owns the emitters. This is intentionally *not* compatible
//! with either container Bloqade Lanes used before (`BLQD`, then `LANES`).
//!
//! [`text`] and [`program`] are the two entry points: `.sst` in and out, and
//! `VHBC` in and out. [`validate`] gates a program against an architecture and
//! type-checks its stack effects.

pub mod bytecode;
pub mod container;
pub mod device;
pub mod machine;
pub mod parse_helpers;
pub mod program;
pub mod resolve;
pub mod syntax;
pub mod text;
pub mod validate;

pub use bytecode::instruction_width;
pub use device::{LanesInstruction, LanesSurfaceInstruction};
pub use machine::{LanesMachine, MachineInstruction, MachineSurfaceInstruction};
pub use program::{LanesInfo, Program, from_code};
pub use text::{parse_text, to_text};
