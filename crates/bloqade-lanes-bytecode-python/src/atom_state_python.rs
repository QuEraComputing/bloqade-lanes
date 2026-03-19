//! PyO3 bindings for AtomStateData.

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use bloqade_lanes_bytecode_core::atom_state::AtomStateData;

use crate::arch_python::{PyArchSpec, PyLaneAddr, PyLocationAddr, PyZoneAddr};

/// Tracks qubit-to-location mappings as atoms move through the architecture.
#[pyclass(name = "AtomStateData", frozen, module = "bloqade.lanes.bytecode")]
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
        locations_to_qubit: Option<HashMap<PyLocationAddr, u32>>,
        qubit_to_locations: Option<HashMap<u32, PyLocationAddr>>,
        collision: Option<HashMap<u32, u32>>,
        prev_lanes: Option<HashMap<u32, PyLaneAddr>>,
        move_count: Option<HashMap<u32, u32>>,
    ) -> Self {
        let inner = AtomStateData {
            locations_to_qubit: locations_to_qubit
                .unwrap_or_default()
                .into_iter()
                .map(|(loc, qubit)| (loc.inner, qubit))
                .collect(),
            qubit_to_locations: qubit_to_locations
                .unwrap_or_default()
                .into_iter()
                .map(|(qubit, loc)| (qubit, loc.inner))
                .collect(),
            collision: collision.unwrap_or_default(),
            prev_lanes: prev_lanes
                .unwrap_or_default()
                .into_iter()
                .map(|(qubit, lane)| (qubit, lane.inner))
                .collect(),
            move_count: move_count.unwrap_or_default(),
        };
        Self { inner }
    }

    /// Create a new state from a mapping of qubit ids to locations.
    #[staticmethod]
    #[pyo3(signature = (locations))]
    fn from_qubit_locations(locations: HashMap<u32, PyLocationAddr>) -> Self {
        let locs: Vec<(u32, LocationAddr)> = locations
            .into_iter()
            .map(|(qubit, loc)| (qubit, loc.inner))
            .collect();
        Self::from_rs(AtomStateData::from_locations(&locs))
    }

    /// Create a new state from a list of locations (qubit ids are 0, 1, 2, ...).
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

    /// Add atoms at new locations. Returns a new state.
    fn add_atoms(&self, locations: HashMap<u32, PyLocationAddr>) -> PyResult<Self> {
        let locs: Vec<(u32, LocationAddr)> = locations
            .into_iter()
            .map(|(qubit, loc)| (qubit, loc.inner))
            .collect();
        self.inner
            .add_atoms(&locs)
            .map(Self::from_rs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Apply lane moves and return a new state, or None if a lane is invalid.
    fn apply_moves(&self, lanes: Vec<PyLaneAddr>, arch_spec: &PyArchSpec) -> Option<Self> {
        let lane_addrs: Vec<LaneAddr> = lanes.iter().map(|l| l.inner).collect();
        self.inner
            .apply_moves(&lane_addrs, &arch_spec.inner)
            .map(Self::from_rs)
    }

    /// Look up the qubit at a given location.
    fn get_qubit(&self, location: &PyLocationAddr) -> Option<u32> {
        self.inner.get_qubit(&location.inner)
    }

    /// Find CZ control/target pairs in a zone.
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
