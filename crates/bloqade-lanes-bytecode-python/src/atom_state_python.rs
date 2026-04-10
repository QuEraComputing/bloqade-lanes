//! PyO3 bindings for AtomStateData.

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use bloqade_lanes_bytecode_core::atom_state::AtomStateData;

use crate::arch_python::{PyArchSpec, PyLaneAddr, PyLocationAddr, PyZoneAddr};
use crate::validation::{validate_i64_key_map, validate_i64_kv_map, validate_i64_value_map};

/// Tracks qubit-to-location mappings as atoms move through the architecture.
///
/// Immutable value type: mutation methods return new instances. Backed by a
/// Rust implementation for performance. Used by the IR analysis pipeline to
/// simulate atom movement, detect collisions, and identify CZ gate pairings.
#[pyclass(name = "AtomStateData", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyAtomStateData {
    pub(crate) inner: AtomStateData,
}

impl PyAtomStateData {
    fn from_rs(inner: AtomStateData) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAtomStateData {
    #[new]
    #[pyo3(signature = (
        locations_to_qubit = None,
        qubit_to_locations = None,
        collision = None,
        prev_lanes = None,
        move_count = None,
    ))]
    fn new(
        locations_to_qubit: Option<HashMap<PyLocationAddr, i64>>,
        qubit_to_locations: Option<HashMap<i64, PyLocationAddr>>,
        collision: Option<HashMap<i64, i64>>,
        prev_lanes: Option<HashMap<i64, PyLaneAddr>>,
        move_count: Option<HashMap<i64, i64>>,
    ) -> PyResult<Self> {
        let locations_to_qubit =
            validate_i64_value_map("qubit_id", locations_to_qubit.unwrap_or_default())?;
        let qubit_to_locations =
            validate_i64_key_map("qubit_id", qubit_to_locations.unwrap_or_default())?;
        let collision = validate_i64_kv_map(
            "qubit_id",
            "collided_qubit_id",
            collision.unwrap_or_default(),
        )?;
        let prev_lanes = validate_i64_key_map("qubit_id", prev_lanes.unwrap_or_default())?;
        let move_count =
            validate_i64_kv_map("qubit_id", "move_count", move_count.unwrap_or_default())?;

        let inner = AtomStateData {
            locations_to_qubit: locations_to_qubit
                .into_iter()
                .map(|(loc, qubit)| (loc.inner, qubit))
                .collect(),
            qubit_to_locations: qubit_to_locations
                .into_iter()
                .map(|(qubit, loc)| (qubit, loc.inner))
                .collect(),
            collision,
            prev_lanes: prev_lanes
                .into_iter()
                .map(|(qubit, lane)| (qubit, lane.inner))
                .collect(),
            move_count,
        };
        Ok(Self { inner })
    }

    /// Create a state from a mapping of qubit ids to locations.
    ///
    /// Collision, prev_lanes, and move_count are initialized to empty.
    /// Qubit ids are validated to fit in u32 range.
    #[staticmethod]
    #[pyo3(signature = (locations))]
    fn from_qubit_locations(locations: HashMap<i64, PyLocationAddr>) -> PyResult<Self> {
        let validated = validate_i64_key_map("qubit_id", locations)?;
        let locs: Vec<(u32, LocationAddr)> = validated
            .into_iter()
            .map(|(qubit, loc)| (qubit, loc.inner))
            .collect();
        Ok(Self::from_rs(AtomStateData::from_locations(&locs)))
    }

    /// Create a state from an ordered list of locations.
    ///
    /// Qubit ids are assigned sequentially starting from 0 based on
    /// list position. Collision, prev_lanes, and move_count are empty.
    #[staticmethod]
    #[pyo3(signature = (locations))]
    fn from_location_list(locations: Vec<PyLocationAddr>) -> Self {
        let locs: Vec<(u32, LocationAddr)> = locations
            .into_iter()
            .enumerate()
            .map(|(i, loc)| (i as u32, loc.inner))
            .collect();
        Self::from_rs(AtomStateData::from_locations(&locs))
    }

    /// Mapping from location to qubit id.
    #[getter]
    fn locations_to_qubit(&self) -> HashMap<PyLocationAddr, u32> {
        self.inner
            .locations_to_qubit
            .iter()
            .map(|(&loc, &qubit)| (PyLocationAddr { inner: loc }, qubit))
            .collect()
    }

    /// Mapping from qubit id to its current location.
    #[getter]
    fn qubit_to_locations(&self) -> HashMap<u32, PyLocationAddr> {
        self.inner
            .qubit_to_locations
            .iter()
            .map(|(&qubit, &loc)| (qubit, PyLocationAddr { inner: loc }))
            .collect()
    }

    /// Mapping from qubit id to another qubit id it collided with.
    #[getter]
    fn collision(&self) -> HashMap<u32, u32> {
        self.inner.collision.clone()
    }

    /// Mapping from qubit id to the lane it took to reach this state.
    #[getter]
    fn prev_lanes(&self) -> HashMap<u32, PyLaneAddr> {
        self.inner
            .prev_lanes
            .iter()
            .map(|(&qubit, &lane)| (qubit, PyLaneAddr { inner: lane }))
            .collect()
    }

    /// Mapping from qubit id to number of moves.
    #[getter]
    fn move_count(&self) -> HashMap<u32, u32> {
        self.inner.move_count.clone()
    }

    /// Add atoms at new locations, returning a new state.
    ///
    /// Qubit ids are validated to fit in u32 range. Raises RuntimeError
    /// if any qubit id already exists or any location is already occupied.
    fn add_atoms(&self, locations: HashMap<i64, PyLocationAddr>) -> PyResult<Self> {
        let validated = validate_i64_key_map("qubit_id", locations)?;
        let locs: Vec<(u32, LocationAddr)> = validated
            .into_iter()
            .map(|(qubit, loc)| (qubit, loc.inner))
            .collect();
        self.inner
            .add_atoms(&locs)
            .map(Self::from_rs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Apply a sequence of lane moves and return the resulting state.
    ///
    /// Each lane is resolved to source/destination locations via the arch spec.
    /// Qubits at source locations are moved to destinations. If a destination
    /// is already occupied, both qubits are recorded as collided and removed
    /// from the location maps. Returns None if any lane is invalid.
    fn apply_moves(&self, lanes: Vec<PyLaneAddr>, arch_spec: &PyArchSpec) -> Option<Self> {
        let lane_addrs: Vec<LaneAddr> = lanes.iter().map(|l| l.inner).collect();
        self.inner
            .apply_moves(&lane_addrs, &arch_spec.inner)
            .map(Self::from_rs)
    }

    /// Look up which qubit (if any) occupies the given location.
    ///
    /// Returns the qubit id, or None if the location is empty.
    fn get_qubit(&self, location: &PyLocationAddr) -> Option<u32> {
        self.inner.get_qubit(&location.inner)
    }

    /// Find CZ gate control/target qubit pairings within a zone.
    ///
    /// Returns (controls, targets, unpaired) where controls[i] and targets[i]
    /// are paired for a CZ gate. Unpaired qubits are those in the zone whose
    /// CZ pair site is empty or doesn't exist. Returns None if the zone is
    /// invalid. Results are sorted by qubit id for deterministic ordering.
    fn get_qubit_pairing(
        &self,
        zone_address: &PyZoneAddr,
        arch_spec: &PyArchSpec,
    ) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let zone = ZoneAddr {
            zone_id: zone_address.inner.zone_id,
        };
        self.inner.get_qubit_pairing(&zone, &arch_spec.inner)
    }

    /// Return a copy of this state.
    fn copy(&self) -> Self {
        self.clone()
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "AtomStateData(qubits={}, collisions={}, moves={})",
            self.inner.qubit_to_locations.len(),
            self.inner.collision.len(),
            self.inner.move_count.len(),
        )
    }
}
