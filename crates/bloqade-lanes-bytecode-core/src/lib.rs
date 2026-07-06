//! # Bloqade Lanes Bytecode Core
//!
//! Pure Rust library for the Bloqade quantum device bytecode format.
//! Provides types and operations for:
//!
//! - **Architecture specification** ([`arch`]) — device topology, transport buses,
//!   zones, grids, and validation
//! - **Bytecode** ([`isa`]) — the instruction set (defined on the
//!   [`vihaco`] virtual-ISA framework), program serialization (native binary +
//!   text SST), and validation
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
//! - [`isa`] — the `Instruction` enum (vihaco-backed), `Program`
//!   (native binary + `.sst` text), and arch-dependent validation

pub mod arch;
pub mod atom_state;
pub mod isa;
pub mod version;

pub use version::Version;
