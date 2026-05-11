//! Architecture specification types, address encoding, and validation.
//!
//! This module defines the physical topology of a Bloqade quantum device:
//! words, grids, transport buses, zones, and the `ArchSpec` that ties them
//! together. It also provides bit-packed address types used by bytecode
//! instructions and comprehensive structural validation.
//!
//! # Key types
//!
//! - [`ArchSpec`] — top-level device specification (loadable from JSON)
//! - [`Word`], [`Grid`], [`Bus`], [`Zone`] — building blocks
//! - [`LocationAddr`], [`LaneAddr`], [`ZoneAddr`] — bit-packed addresses
//! - [`Direction`], [`MoveType`] — transport enums

pub mod addr;
pub mod query;
pub mod types;
pub mod validate;

pub use addr::{
    Direction, LaneAddr, LocationAddr, MoveType, SiteRef, WordRef, ZoneAddr, ZonedWordRef,
};
pub use query::ArchSpecLoadError;
pub use types::{ArchSpec, Bus, Grid, Mode, TransportPath, Word, Zone};
pub use validate::ArchSpecError;
