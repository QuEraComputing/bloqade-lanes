//! # Bloqade Lanes Bytecode Core
//!
//! Pure Rust library for the Bloqade quantum device bytecode format.
//! Provides types and operations for:
//!
//! - **Architecture specification** ([`arch`]) — device topology, transport buses,
//!   zones, grids, and validation
//! - **Bytecode** ([`bytecode`]) — instruction encoding/decoding, program
//!   serialization (binary BLQD + text SST), and validation
//! - **Versioning** ([`Version`]) — semantic versioning for arch specs and programs
//!
//! This crate contains no Python or C FFI dependencies. It is the foundation
//! that the PyO3 bindings and CLI tool build upon.
//!
//! ## Crate layout
//!
//! - [`arch::types`] — `ArchSpec`, `Word`, `Grid`, `Bus`, `Zone`, etc.
//! - [`arch::addr`] — bit-packed address types (`LocationAddr`, `LaneAddr`, `ZoneAddr`)
//! - [`arch::query`] — arch spec queries (position lookup, lane resolution, JSON loading)
//! - [`arch::validate`] — structural validation with collected errors
//! - [`bytecode::instruction`] — instruction enum and opcode computation
//! - [`bytecode::opcode`] — device codes, instruction codes, opcode packing
//! - [`bytecode::encode`] — binary encoding/decoding of instructions
//! - [`bytecode::program`] — `Program` type with binary (BLQD) serialization
//! - [`bytecode::text`] — SST text assembly format (parse/print)
//! - [`bytecode::validate`] — program validation (structural, address, stack simulation)

pub mod arch;
pub mod bytecode;
pub mod version;

pub use version::Version;
