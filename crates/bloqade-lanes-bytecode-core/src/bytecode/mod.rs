//! Bytecode instruction set, program format, and validation.
//!
//! Every instruction is a fixed 16-byte word: a 32-bit opcode followed by
//! three 32-bit data words, all little-endian. The opcode packs a device
//! code (low byte) and instruction code (next byte).
//!
//! # Modules
//!
//! - [`instruction`] — high-level instruction enum
//! - [`opcode`] — device codes, instruction codes, opcode packing/decoding
//! - [`encode`] — binary encoding/decoding of instructions to/from 16-byte words
//! - [`program`] — `Program` type with BLQD binary serialization
//! - [`text`] — SST text assembly format (human-readable parse/print)
//! - [`validate`] — structural, address, and stack-simulation validation

pub mod encode;
pub mod instruction;
pub mod opcode;
pub mod program;
pub mod text;
pub mod validate;
pub mod value;

pub use crate::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType, ZoneAddr};
pub use encode::DecodeError;
pub use instruction::{
    ArrayInstruction, AtomArrangementInstruction, CpuInstruction, DetectorObservableInstruction,
    Instruction, LaneConstInstruction, MeasurementInstruction, QuantumGateInstruction,
};
pub use opcode::DeviceCode;
pub use program::{Program, ProgramError};
pub use text::ParseError;
pub use validate::ValidationError;
pub use value::{ArrayValue, CpuValue, DeviceValue, Value};
