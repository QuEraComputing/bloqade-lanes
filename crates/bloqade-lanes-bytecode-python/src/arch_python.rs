use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr as rs_addr;
use bloqade_lanes_bytecode_core::arch::types as rs;
use bloqade_lanes_bytecode_core::version::Version;

/// Validate that a field value fits in 16 bits (0..=65535).
fn validate_u16_field(name: &str, value: i64) -> PyResult<u32> {
    if value < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}={} must be non-negative",
            name, value
        )));
    }
    if value > 0xFFFF {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}={} exceeds maximum 65535",
            name, value
        )));
    }
    Ok(value as u32)
}

// ── Direction enum ──

#[pyclass(
    name = "Direction",
    eq,
    eq_int,
    hash,
    frozen,
    module = "bloqade.lanes.bytecode"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PyDirection {
    #[pyo3(name = "FORWARD")]
    Forward = 0,
    #[pyo3(name = "BACKWARD")]
    Backward = 1,
}

#[pymethods]
impl PyDirection {
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            PyDirection::Forward => "FORWARD",
            PyDirection::Backward => "BACKWARD",
        }
    }
}

impl PyDirection {
    pub(crate) fn from_rs(d: rs_addr::Direction) -> Self {
        match d {
            rs_addr::Direction::Forward => PyDirection::Forward,
            rs_addr::Direction::Backward => PyDirection::Backward,
        }
    }

    pub(crate) fn to_rs(&self) -> rs_addr::Direction {
        match self {
            PyDirection::Forward => rs_addr::Direction::Forward,
            PyDirection::Backward => rs_addr::Direction::Backward,
        }
    }
}

// ── MoveType enum ──

#[pyclass(
    name = "MoveType",
    eq,
    eq_int,
    hash,
    frozen,
    module = "bloqade.lanes.bytecode"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PyMoveType {
    #[pyo3(name = "SITE")]
    SiteBus = 0,
    #[pyo3(name = "WORD")]
    WordBus = 1,
}

#[pymethods]
impl PyMoveType {
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            PyMoveType::SiteBus => "SITE",
            PyMoveType::WordBus => "WORD",
        }
    }
}

impl PyMoveType {
    pub(crate) fn from_rs(m: rs_addr::MoveType) -> Self {
        match m {
            rs_addr::MoveType::SiteBus => PyMoveType::SiteBus,
            rs_addr::MoveType::WordBus => PyMoveType::WordBus,
        }
    }

    pub(crate) fn to_rs(&self) -> rs_addr::MoveType {
        match self {
            PyMoveType::SiteBus => rs_addr::MoveType::SiteBus,
            PyMoveType::WordBus => rs_addr::MoveType::WordBus,
        }
    }
}

// ── LocationAddr ──

#[pyclass(name = "LocationAddress", frozen, module = "bloqade.lanes.bytecode")]
#[derive(Clone)]
pub struct PyLocationAddr {
    pub(crate) inner: rs_addr::LocationAddr,
}

#[pymethods]
impl PyLocationAddr {
    #[new]
    fn new(word_id: i64, site_id: i64) -> PyResult<Self> {
        let word_id = validate_u16_field("word_id", word_id)?;
        let site_id = validate_u16_field("site_id", site_id)?;
        Ok(Self {
            inner: rs_addr::LocationAddr { word_id, site_id },
        })
    }

    #[getter]
    fn word_id(&self) -> u32 {
        self.inner.word_id
    }

    #[getter]
    fn site_id(&self) -> u32 {
        self.inner.site_id
    }

    fn encode(&self) -> u32 {
        self.inner.encode()
    }

    #[staticmethod]
    fn decode(bits: u32) -> Self {
        Self {
            inner: rs_addr::LocationAddr::decode(bits),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LocationAddress(word_id={}, site_id={})",
            self.inner.word_id, self.inner.site_id
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        self.inner.encode() as u64
    }
}

// ── LaneAddr ──

#[pyclass(name = "LaneAddress", frozen, module = "bloqade.lanes.bytecode")]
#[derive(Clone)]
pub struct PyLaneAddr {
    pub(crate) inner: rs_addr::LaneAddr,
}

#[pymethods]
impl PyLaneAddr {
    #[new]
    #[pyo3(signature = (move_type, word_id, site_id, bus_id, direction=PyDirection::Forward))]
    fn new(
        move_type: &PyMoveType,
        word_id: i64,
        site_id: i64,
        bus_id: i64,
        direction: PyDirection,
    ) -> PyResult<Self> {
        let word_id = validate_u16_field("word_id", word_id)?;
        let site_id = validate_u16_field("site_id", site_id)?;
        let bus_id = validate_u16_field("bus_id", bus_id)?;
        Ok(Self {
            inner: rs_addr::LaneAddr {
                direction: direction.to_rs(),
                move_type: move_type.to_rs(),
                word_id,
                site_id,
                bus_id,
            },
        })
    }

    #[getter]
    fn direction(&self) -> PyDirection {
        PyDirection::from_rs(self.inner.direction)
    }

    #[getter]
    fn move_type(&self) -> PyMoveType {
        PyMoveType::from_rs(self.inner.move_type)
    }

    #[getter]
    fn word_id(&self) -> u32 {
        self.inner.word_id
    }

    #[getter]
    fn site_id(&self) -> u32 {
        self.inner.site_id
    }

    #[getter]
    fn bus_id(&self) -> u32 {
        self.inner.bus_id
    }

    fn encode(&self) -> u64 {
        let (d0, d1) = self.inner.encode();
        (d0 as u64) | ((d1 as u64) << 32)
    }

    #[staticmethod]
    fn decode(bits: u64) -> Self {
        let d0 = bits as u32;
        let d1 = (bits >> 32) as u32;
        Self {
            inner: rs_addr::LaneAddr::decode(d0, d1),
        }
    }

    fn __repr__(&self) -> String {
        let dir = match self.inner.direction {
            rs_addr::Direction::Forward => "Direction.FORWARD",
            rs_addr::Direction::Backward => "Direction.BACKWARD",
        };
        let mt = match self.inner.move_type {
            rs_addr::MoveType::SiteBus => "MoveType.SITE",
            rs_addr::MoveType::WordBus => "MoveType.WORD",
        };
        format!(
            "LaneAddress(move_type={}, word_id={}, site_id={}, bus_id={}, direction={})",
            mt, self.inner.word_id, self.inner.site_id, self.inner.bus_id, dir
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let (d0, d1) = self.inner.encode();
        (d0 as u64) | ((d1 as u64) << 32)
    }
}

// ── ZoneAddr ──

#[pyclass(name = "ZoneAddress", frozen, module = "bloqade.lanes.bytecode")]
#[derive(Clone)]
pub struct PyZoneAddr {
    pub(crate) inner: rs_addr::ZoneAddr,
}

#[pymethods]
impl PyZoneAddr {
    #[new]
    fn new(zone_id: i64) -> PyResult<Self> {
        let zone_id = validate_u16_field("zone_id", zone_id)?;
        Ok(Self {
            inner: rs_addr::ZoneAddr { zone_id },
        })
    }

    #[getter]
    fn zone_id(&self) -> u32 {
        self.inner.zone_id
    }

    fn encode(&self) -> u32 {
        self.inner.encode()
    }

    #[staticmethod]
    fn decode(bits: u32) -> Self {
        Self {
            inner: rs_addr::ZoneAddr::decode(bits),
        }
    }

    fn __repr__(&self) -> String {
        format!("ZoneAddress(zone_id={})", self.inner.zone_id)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        self.inner.encode() as u64
    }
}

// ── Grid ──

#[pyclass(name = "Grid", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyGrid {
    pub(crate) inner: rs::Grid,
}

#[pymethods]
impl PyGrid {
    #[new]
    fn new(x_start: f64, y_start: f64, x_spacing: Vec<f64>, y_spacing: Vec<f64>) -> Self {
        Self {
            inner: rs::Grid {
                x_start,
                y_start,
                x_spacing,
                y_spacing,
            },
        }
    }

    /// Construct a Grid from explicit position arrays.
    ///
    /// The first element becomes the start value and consecutive differences
    /// become the spacing vector.
    #[classmethod]
    fn from_positions(
        _cls: &Bound<'_, pyo3::types::PyType>,
        x_positions: Vec<f64>,
        y_positions: Vec<f64>,
    ) -> PyResult<Self> {
        if x_positions.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "x_positions must have at least one element",
            ));
        }
        if y_positions.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "y_positions must have at least one element",
            ));
        }
        Ok(Self {
            inner: rs::Grid::from_positions(&x_positions, &y_positions),
        })
    }

    #[getter]
    fn num_x(&self) -> usize {
        self.inner.num_x()
    }

    #[getter]
    fn num_y(&self) -> usize {
        self.inner.num_y()
    }

    #[getter]
    fn x_start(&self) -> f64 {
        self.inner.x_start
    }

    #[getter]
    fn y_start(&self) -> f64 {
        self.inner.y_start
    }

    #[getter]
    fn x_spacing(&self) -> Vec<f64> {
        self.inner.x_spacing.clone()
    }

    #[getter]
    fn y_spacing(&self) -> Vec<f64> {
        self.inner.y_spacing.clone()
    }

    /// Compute all x-coordinates from start + cumulative spacing.
    #[getter]
    fn x_positions(&self) -> Vec<f64> {
        self.inner.x_positions()
    }

    /// Compute all y-coordinates from start + cumulative spacing.
    #[getter]
    fn y_positions(&self) -> Vec<f64> {
        self.inner.y_positions()
    }

    fn __repr__(&self) -> String {
        format!(
            "Grid(x_start={}, y_start={}, x_spacing={:?}, y_spacing={:?})",
            self.inner.x_start, self.inner.y_start, self.inner.x_spacing, self.inner.y_spacing
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.x_start.to_bits().hash(&mut hasher);
        self.inner.y_start.to_bits().hash(&mut hasher);
        for v in &self.inner.x_spacing {
            v.to_bits().hash(&mut hasher);
        }
        for v in &self.inner.y_spacing {
            v.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

// ── Word ──

#[pyclass(name = "Word", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyWord {
    pub(crate) inner: rs::Word,
}

#[pymethods]
impl PyWord {
    #[new]
    #[pyo3(signature = (grid, sites, cz_pairs=None))]
    fn new(grid: &PyGrid, sites: Vec<(u32, u32)>, cz_pairs: Option<Vec<(u32, u32)>>) -> Self {
        Self {
            inner: rs::Word {
                grid: grid.inner.clone(),
                sites: sites.into_iter().map(|(x, y)| [x, y]).collect(),
                cz_pairs: cz_pairs.map(|v| v.into_iter().map(|(a, b)| [a, b]).collect()),
            },
        }
    }

    #[getter]
    fn grid(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.grid.clone(),
        }
    }

    #[getter]
    fn sites(&self) -> Vec<(u32, u32)> {
        self.inner.sites.iter().map(|s| (s[0], s[1])).collect()
    }

    #[getter]
    fn cz_pairs(&self) -> Option<Vec<(u32, u32)>> {
        self.inner
            .cz_pairs
            .as_ref()
            .map(|v| v.iter().map(|p| (p[0], p[1])).collect())
    }

    fn site_position(&self, site_idx: usize) -> Option<(f64, f64)> {
        self.inner.site_position(site_idx)
    }

    fn __repr__(&self) -> String {
        format!("Word(sites={})", self.inner.sites.len())
    }
}

// ── Geometry ──

#[pyclass(name = "Geometry", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyGeometry {
    pub(crate) inner: rs::Geometry,
}

#[pymethods]
impl PyGeometry {
    #[new]
    fn new(sites_per_word: u32, words: Vec<PyRef<'_, PyWord>>) -> Self {
        Self {
            inner: rs::Geometry {
                sites_per_word,
                words: words.iter().map(|w| w.inner.clone()).collect(),
            },
        }
    }

    #[getter]
    fn sites_per_word(&self) -> u32 {
        self.inner.sites_per_word
    }

    #[getter]
    fn words(&self) -> Vec<PyWord> {
        self.inner
            .words
            .iter()
            .map(|w| PyWord { inner: w.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Geometry(sites_per_word={}, words={})",
            self.inner.sites_per_word,
            self.inner.words.len()
        )
    }
}

// ── Bus ──

#[pyclass(name = "Bus", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyBus {
    pub(crate) inner: rs::Bus,
}

#[pymethods]
impl PyBus {
    #[new]
    fn new(src: Vec<u32>, dst: Vec<u32>) -> Self {
        Self {
            inner: rs::Bus { src, dst },
        }
    }

    #[getter]
    fn src(&self) -> Vec<u32> {
        self.inner.src.clone()
    }

    #[getter]
    fn dst(&self) -> Vec<u32> {
        self.inner.dst.clone()
    }

    /// Map a source value to its destination (forward move).
    /// Returns None if not found.
    fn resolve_forward(&self, src: u32) -> Option<u32> {
        self.inner.resolve_forward(src)
    }

    /// Map a destination value back to its source (backward move).
    /// Returns None if not found.
    fn resolve_backward(&self, dst: u32) -> Option<u32> {
        self.inner.resolve_backward(dst)
    }

    fn __repr__(&self) -> String {
        format!("Bus(src={:?}, dst={:?})", self.inner.src, self.inner.dst)
    }
}

// ── Buses ──

#[pyclass(name = "Buses", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyBuses {
    pub(crate) inner: rs::Buses,
}

#[pymethods]
impl PyBuses {
    #[new]
    fn new(site_buses: Vec<PyRef<'_, PyBus>>, word_buses: Vec<PyRef<'_, PyBus>>) -> Self {
        Self {
            inner: rs::Buses {
                site_buses: site_buses.iter().map(|b| b.inner.clone()).collect(),
                word_buses: word_buses.iter().map(|b| b.inner.clone()).collect(),
            },
        }
    }

    #[getter]
    fn site_buses(&self) -> Vec<PyBus> {
        self.inner
            .site_buses
            .iter()
            .map(|b| PyBus { inner: b.clone() })
            .collect()
    }

    #[getter]
    fn word_buses(&self) -> Vec<PyBus> {
        self.inner
            .word_buses
            .iter()
            .map(|b| PyBus { inner: b.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Buses(site_buses={}, word_buses={})",
            self.inner.site_buses.len(),
            self.inner.word_buses.len()
        )
    }
}

// ── Zone ──

#[pyclass(name = "Zone", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyZone {
    pub(crate) inner: rs::Zone,
}

#[pymethods]
impl PyZone {
    #[new]
    fn new(words: Vec<u32>) -> Self {
        Self {
            inner: rs::Zone { words },
        }
    }

    #[getter]
    fn words(&self) -> Vec<u32> {
        self.inner.words.clone()
    }

    fn __repr__(&self) -> String {
        format!("Zone(words={:?})", self.inner.words)
    }
}

// ── TransportPath ──

#[pyclass(name = "TransportPath", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyTransportPath {
    pub(crate) inner: rs::TransportPath,
}

#[pymethods]
impl PyTransportPath {
    #[new]
    fn new(lane: &PyLaneAddr, waypoints: Vec<(f64, f64)>) -> Self {
        let (d0, d1) = lane.inner.encode();
        Self {
            inner: rs::TransportPath {
                lane: (d0 as u64) | ((d1 as u64) << 32),
                waypoints: waypoints.into_iter().map(|(x, y)| [x, y]).collect(),
            },
        }
    }

    #[getter]
    fn lane(&self) -> PyLaneAddr {
        let d0 = self.inner.lane as u32;
        let d1 = (self.inner.lane >> 32) as u32;
        PyLaneAddr {
            inner: rs_addr::LaneAddr::decode(d0, d1),
        }
    }

    #[getter]
    fn lane_encoded(&self) -> u64 {
        self.inner.lane
    }

    #[getter]
    fn waypoints(&self) -> Vec<(f64, f64)> {
        self.inner.waypoints.iter().map(|w| (w[0], w[1])).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TransportPath(lane=0x{:08X}, waypoints={})",
            self.inner.lane,
            self.inner.waypoints.len()
        )
    }
}

// ── ArchSpec ──

#[pyclass(name = "ArchSpec", frozen, module = "bloqade.lanes.bytecode.arch")]
#[derive(Clone)]
pub struct PyArchSpec {
    pub(crate) inner: rs::ArchSpec,
}

#[pymethods]
impl PyArchSpec {
    #[new]
    #[pyo3(signature = (version, geometry, buses, words_with_site_buses, sites_with_word_buses, zones, entangling_zones, measurement_mode_zones, paths=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        version: (u16, u16),
        geometry: &PyGeometry,
        buses: &PyBuses,
        words_with_site_buses: Vec<u32>,
        sites_with_word_buses: Vec<u32>,
        zones: Vec<PyRef<'_, PyZone>>,
        entangling_zones: Vec<u32>,
        measurement_mode_zones: Vec<u32>,
        paths: Option<Vec<PyRef<'_, PyTransportPath>>>,
    ) -> Self {
        Self {
            inner: rs::ArchSpec {
                version: Version::new(version.0, version.1),
                geometry: geometry.inner.clone(),
                buses: buses.inner.clone(),
                words_with_site_buses,
                sites_with_word_buses,
                zones: zones.iter().map(|z| z.inner.clone()).collect(),
                entangling_zones,
                measurement_mode_zones,
                paths: paths.map(|v| v.iter().map(|p| p.inner.clone()).collect()),
            },
        }
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = rs::ArchSpec::from_json(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_json_validated(json: &str, py: Python<'_>) -> PyResult<Self> {
        let inner = rs::ArchSpec::from_json_validated(json)
            .map_err(|e| crate::errors::arch_spec_load_error_to_py(py, &e))?;
        Ok(Self { inner })
    }

    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|errors| crate::errors::arch_spec_errors_to_py(py, errors))
    }

    #[getter]
    fn version(&self) -> (u16, u16) {
        (self.inner.version.major, self.inner.version.minor)
    }

    #[getter]
    fn geometry(&self) -> PyGeometry {
        PyGeometry {
            inner: self.inner.geometry.clone(),
        }
    }

    #[getter]
    fn buses(&self) -> PyBuses {
        PyBuses {
            inner: self.inner.buses.clone(),
        }
    }

    #[getter]
    fn words_with_site_buses(&self) -> Vec<u32> {
        self.inner.words_with_site_buses.clone()
    }

    #[getter]
    fn sites_with_word_buses(&self) -> Vec<u32> {
        self.inner.sites_with_word_buses.clone()
    }

    #[getter]
    fn zones(&self) -> Vec<PyZone> {
        self.inner
            .zones
            .iter()
            .map(|z| PyZone { inner: z.clone() })
            .collect()
    }

    #[getter]
    fn entangling_zones(&self) -> Vec<u32> {
        self.inner.entangling_zones.clone()
    }

    #[getter]
    fn measurement_mode_zones(&self) -> Vec<u32> {
        self.inner.measurement_mode_zones.clone()
    }

    #[getter]
    fn paths(&self) -> Option<Vec<PyTransportPath>> {
        self.inner.paths.as_ref().map(|v| {
            v.iter()
                .map(|p| PyTransportPath { inner: p.clone() })
                .collect()
        })
    }

    fn word_by_id(&self, id: u32) -> Option<PyWord> {
        self.inner
            .word_by_id(id)
            .map(|w| PyWord { inner: w.clone() })
    }

    fn zone_by_id(&self, id: u32) -> Option<PyZone> {
        self.inner
            .zone_by_id(id)
            .map(|z| PyZone { inner: z.clone() })
    }

    /// Look up a site bus by its identifier. Returns None if not found.
    fn site_bus_by_id(&self, id: u32) -> Option<PyBus> {
        self.inner
            .site_bus_by_id(id)
            .map(|b| PyBus { inner: b.clone() })
    }

    /// Look up a word bus by its identifier. Returns None if not found.
    fn word_bus_by_id(&self, id: u32) -> Option<PyBus> {
        self.inner
            .word_bus_by_id(id)
            .map(|b| PyBus { inner: b.clone() })
    }

    /// Resolve a location address to its physical (x, y) coordinates.
    ///
    /// Returns None if the word or site is not found in the geometry.
    #[pyo3(text_signature = "(self, loc)")]
    fn location_position(&self, loc: &PyLocationAddr) -> Option<(f64, f64)> {
        self.inner.location_position(&loc.inner)
    }

    /// Resolve a lane address to its source and destination location addresses.
    ///
    /// Given a ``LaneAddr``, determines which two ``LocationAddr`` endpoints the
    /// lane connects by tracing through the appropriate bus (site bus or word bus)
    /// in the specified direction (forward or backward).
    ///
    /// Returns a ``(src, dst)`` tuple of ``LocationAddr``, or None if the lane
    /// references an invalid bus, word, or site.
    #[pyo3(text_signature = "(self, lane)")]
    fn lane_endpoints(&self, lane: &PyLaneAddr) -> Option<(PyLocationAddr, PyLocationAddr)> {
        let (src, dst) = self.inner.lane_endpoints(&lane.inner)?;
        Some((PyLocationAddr { inner: src }, PyLocationAddr { inner: dst }))
    }

    fn check_zone(&self, addr: &PyZoneAddr) -> Option<String> {
        self.inner.check_zone(&addr.inner)
    }

    fn check_locations(
        &self,
        py: Python<'_>,
        locations: Vec<PyRef<'_, PyLocationAddr>>,
    ) -> PyResult<PyObject> {
        let addrs: Vec<rs_addr::LocationAddr> = locations.iter().map(|l| l.inner).collect();
        let errors = self.inner.check_locations(&addrs);
        let py_errors: Vec<PyObject> = errors
            .iter()
            .map(|e| crate::errors::location_group_error_to_py(py, e))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(pyo3::types::PyList::new(py, &py_errors)?.into())
    }

    fn check_lanes(&self, py: Python<'_>, lanes: Vec<PyRef<'_, PyLaneAddr>>) -> PyResult<PyObject> {
        let addrs: Vec<rs_addr::LaneAddr> = lanes.iter().map(|l| l.inner).collect();
        let errors = self.inner.check_lanes(&addrs);
        let py_errors: Vec<PyObject> = errors
            .iter()
            .map(|e| crate::errors::lane_group_error_to_py(py, e))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(pyo3::types::PyList::new(py, &py_errors)?.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "ArchSpec(version=({}, {}))",
            self.inner.version.major, self.inner.version.minor
        )
    }
}
