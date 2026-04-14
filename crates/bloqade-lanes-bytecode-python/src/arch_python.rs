use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr as rs_addr;
use bloqade_lanes_bytecode_core::arch::types as rs;
use bloqade_lanes_bytecode_core::version::Version;

use crate::validation::{validate_field, validate_vec};

// ── Direction enum ──

#[pyclass(
    name = "Direction",
    eq,
    eq_int,
    hash,
    frozen,
    module = "bloqade.lanes.bytecode._native"
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
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PyMoveType {
    #[pyo3(name = "SITE")]
    SiteBus = 0,
    #[pyo3(name = "WORD")]
    WordBus = 1,
    #[pyo3(name = "ZONE")]
    ZoneBus = 2,
}

#[pymethods]
impl PyMoveType {
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            PyMoveType::SiteBus => "SITE",
            PyMoveType::WordBus => "WORD",
            PyMoveType::ZoneBus => "ZONE",
        }
    }
}

impl PyMoveType {
    pub(crate) fn from_rs(m: rs_addr::MoveType) -> Self {
        match m {
            rs_addr::MoveType::SiteBus => PyMoveType::SiteBus,
            rs_addr::MoveType::WordBus => PyMoveType::WordBus,
            rs_addr::MoveType::ZoneBus => PyMoveType::ZoneBus,
        }
    }

    pub(crate) fn to_rs(&self) -> rs_addr::MoveType {
        match self {
            PyMoveType::SiteBus => rs_addr::MoveType::SiteBus,
            PyMoveType::WordBus => rs_addr::MoveType::WordBus,
            PyMoveType::ZoneBus => rs_addr::MoveType::ZoneBus,
        }
    }
}

// ── LocationAddr ──

#[pyclass(
    name = "LocationAddress",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyLocationAddr {
    pub(crate) inner: rs_addr::LocationAddr,
}

#[pymethods]
impl PyLocationAddr {
    #[new]
    fn new(zone_id: i64, word_id: i64, site_id: i64) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
        let word_id = validate_field::<u16>("word_id", word_id)? as u32;
        let site_id = validate_field::<u16>("site_id", site_id)? as u32;
        Ok(Self {
            inner: rs_addr::LocationAddr {
                zone_id,
                word_id,
                site_id,
            },
        })
    }

    #[getter]
    fn zone_id(&self) -> u32 {
        self.inner.zone_id
    }

    #[getter]
    fn word_id(&self) -> u32 {
        self.inner.word_id
    }

    #[getter]
    fn site_id(&self) -> u32 {
        self.inner.site_id
    }

    fn encode(&self) -> u64 {
        self.inner.encode()
    }

    #[staticmethod]
    fn decode(bits: u64) -> Self {
        Self {
            inner: rs_addr::LocationAddr::decode(bits),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LocationAddress(zone_id={}, word_id={}, site_id={})",
            self.inner.zone_id, self.inner.word_id, self.inner.site_id
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        self.inner.encode()
    }
}

// ── LaneAddr ──

#[pyclass(
    name = "LaneAddress",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyLaneAddr {
    pub(crate) inner: rs_addr::LaneAddr,
}

#[pymethods]
impl PyLaneAddr {
    #[new]
    #[pyo3(signature = (move_type, zone_id, word_id, site_id, bus_id, direction=PyDirection::Forward))]
    fn new(
        move_type: &PyMoveType,
        zone_id: i64,
        word_id: i64,
        site_id: i64,
        bus_id: i64,
        direction: PyDirection,
    ) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
        let word_id = validate_field::<u16>("word_id", word_id)? as u32;
        let site_id = validate_field::<u16>("site_id", site_id)? as u32;
        let bus_id = validate_field::<u16>("bus_id", bus_id)? as u32;
        Ok(Self {
            inner: rs_addr::LaneAddr {
                direction: direction.to_rs(),
                move_type: move_type.to_rs(),
                zone_id,
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
    fn zone_id(&self) -> u32 {
        self.inner.zone_id
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
        self.inner.encode_u64()
    }

    #[staticmethod]
    fn decode(bits: u64) -> Self {
        Self {
            inner: rs_addr::LaneAddr::decode_u64(bits),
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
            rs_addr::MoveType::ZoneBus => "MoveType.ZONE",
        };
        format!(
            "LaneAddress(move_type={}, zone_id={}, word_id={}, site_id={}, bus_id={}, direction={})",
            mt, self.inner.zone_id, self.inner.word_id, self.inner.site_id, self.inner.bus_id, dir
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        self.inner.encode_u64()
    }
}

// ── ZoneAddr ──

#[pyclass(
    name = "ZoneAddress",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyZoneAddr {
    pub(crate) inner: rs_addr::ZoneAddr,
}

#[pymethods]
impl PyZoneAddr {
    #[new]
    fn new(zone_id: i64) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
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

#[pyclass(name = "Grid", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyGrid {
    pub(crate) inner: rs::Grid,
}

#[pymethods]
impl PyGrid {
    #[new]
    fn new(x_start: f64, y_start: f64, x_spacing: Vec<f64>, y_spacing: Vec<f64>) -> PyResult<Self> {
        let grid = rs::Grid {
            x_start,
            y_start,
            x_spacing,
            y_spacing,
        };
        if let Err(field) = grid.check_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{field} contains non-finite value (NaN or Inf)"
            )));
        }
        Ok(Self { inner: grid })
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
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ── Word ──

#[pyclass(name = "Word", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyWord {
    pub(crate) inner: rs::Word,
}

#[pymethods]
impl PyWord {
    #[new]
    fn new(sites: Vec<(u32, u32)>) -> Self {
        Self {
            inner: rs::Word {
                sites: sites.into_iter().map(|(x, y)| [x, y]).collect(),
            },
        }
    }

    #[getter]
    fn sites(&self) -> Vec<(u32, u32)> {
        self.inner.sites.iter().map(|s| (s[0], s[1])).collect()
    }

    fn __repr__(&self) -> String {
        format!("Word(sites={})", self.inner.sites.len())
    }
}

// ── SiteBus ──

#[pyclass(name = "SiteBus", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PySiteBus {
    pub(crate) inner: rs::Bus<rs_addr::SiteRef>,
}

#[pymethods]
impl PySiteBus {
    #[new]
    fn new(src: Vec<u16>, dst: Vec<u16>) -> Self {
        Self {
            inner: rs::Bus {
                src: src.into_iter().map(rs_addr::SiteRef).collect(),
                dst: dst.into_iter().map(rs_addr::SiteRef).collect(),
            },
        }
    }

    #[getter]
    fn src(&self) -> Vec<u16> {
        self.inner.src.iter().map(|s| s.0).collect()
    }

    #[getter]
    fn dst(&self) -> Vec<u16> {
        self.inner.dst.iter().map(|d| d.0).collect()
    }

    /// Map a source value to its destination (forward move).
    /// Returns None if not found.
    fn resolve_forward(&self, src: u16) -> Option<u16> {
        self.inner.resolve_forward(src)
    }

    /// Map a destination value back to its source (backward move).
    /// Returns None if not found.
    fn resolve_backward(&self, dst: u16) -> Option<u16> {
        self.inner.resolve_backward(dst)
    }

    fn __repr__(&self) -> String {
        let src: Vec<u16> = self.inner.src.iter().map(|s| s.0).collect();
        let dst: Vec<u16> = self.inner.dst.iter().map(|d| d.0).collect();
        format!("SiteBus(src={:?}, dst={:?})", src, dst)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ── WordBus ──

#[pyclass(name = "WordBus", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyWordBus {
    pub(crate) inner: rs::Bus<rs_addr::WordRef>,
}

#[pymethods]
impl PyWordBus {
    #[new]
    fn new(src: Vec<u16>, dst: Vec<u16>) -> Self {
        Self {
            inner: rs::Bus {
                src: src.into_iter().map(rs_addr::WordRef).collect(),
                dst: dst.into_iter().map(rs_addr::WordRef).collect(),
            },
        }
    }

    #[getter]
    fn src(&self) -> Vec<u16> {
        self.inner.src.iter().map(|s| s.0).collect()
    }

    #[getter]
    fn dst(&self) -> Vec<u16> {
        self.inner.dst.iter().map(|d| d.0).collect()
    }

    /// Map a source value to its destination (forward move).
    /// Returns None if not found.
    fn resolve_forward(&self, src: u16) -> Option<u16> {
        self.inner.resolve_forward(src)
    }

    /// Map a destination value back to its source (backward move).
    /// Returns None if not found.
    fn resolve_backward(&self, dst: u16) -> Option<u16> {
        self.inner.resolve_backward(dst)
    }

    fn __repr__(&self) -> String {
        let src: Vec<u16> = self.inner.src.iter().map(|s| s.0).collect();
        let dst: Vec<u16> = self.inner.dst.iter().map(|d| d.0).collect();
        format!("WordBus(src={:?}, dst={:?})", src, dst)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ── ZoneBus (inter-zone) ──

#[pyclass(name = "ZoneBus", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyZoneBus {
    pub(crate) inner: rs::Bus<rs_addr::ZonedWordRef>,
}

#[pymethods]
impl PyZoneBus {
    #[new]
    fn new(src: Vec<(u8, u16)>, dst: Vec<(u8, u16)>) -> Self {
        Self {
            inner: rs::Bus {
                src: src
                    .into_iter()
                    .map(|(z, w)| rs_addr::ZonedWordRef {
                        zone_id: z,
                        word_id: w,
                    })
                    .collect(),
                dst: dst
                    .into_iter()
                    .map(|(z, w)| rs_addr::ZonedWordRef {
                        zone_id: z,
                        word_id: w,
                    })
                    .collect(),
            },
        }
    }

    #[getter]
    fn src(&self) -> Vec<(u8, u16)> {
        self.inner
            .src
            .iter()
            .map(|s| (s.zone_id, s.word_id))
            .collect()
    }

    #[getter]
    fn dst(&self) -> Vec<(u8, u16)> {
        self.inner
            .dst
            .iter()
            .map(|d| (d.zone_id, d.word_id))
            .collect()
    }

    fn __repr__(&self) -> String {
        let src: Vec<(u8, u16)> = self
            .inner
            .src
            .iter()
            .map(|s| (s.zone_id, s.word_id))
            .collect();
        let dst: Vec<(u8, u16)> = self
            .inner
            .dst
            .iter()
            .map(|d| (d.zone_id, d.word_id))
            .collect();
        format!("ZoneBus(src={:?}, dst={:?})", src, dst)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ── Zone ──

#[pyclass(name = "Zone", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyZone {
    pub(crate) inner: rs::Zone,
}

#[pymethods]
impl PyZone {
    #[new]
    #[pyo3(signature = (name, grid, site_buses, word_buses, words_with_site_buses, sites_with_word_buses, entangling_pairs=None))]
    fn new(
        name: String,
        grid: &PyGrid,
        site_buses: Vec<PyRef<'_, PySiteBus>>,
        word_buses: Vec<PyRef<'_, PyWordBus>>,
        words_with_site_buses: Vec<i64>,
        sites_with_word_buses: Vec<i64>,
        entangling_pairs: Option<Vec<(i64, i64)>>,
    ) -> PyResult<Self> {
        let words_with_site_buses =
            validate_vec::<u32>("words_with_site_buses", words_with_site_buses)?;
        let sites_with_word_buses =
            validate_vec::<u32>("sites_with_word_buses", sites_with_word_buses)?;
        let entangling_pairs: Vec<[u32; 2]> = entangling_pairs
            .unwrap_or_default()
            .into_iter()
            .map(|(a, b)| {
                let a = validate_field::<u32>("entangling_pairs word_id", a)?;
                let b = validate_field::<u32>("entangling_pairs word_id", b)?;
                Ok([a, b])
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: rs::Zone {
                name,
                grid: grid.inner.clone(),
                site_buses: site_buses.iter().map(|b| b.inner.clone()).collect(),
                word_buses: word_buses.iter().map(|b| b.inner.clone()).collect(),
                words_with_site_buses,
                sites_with_word_buses,
                entangling_pairs,
            },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn grid(&self) -> PyGrid {
        PyGrid {
            inner: self.inner.grid.clone(),
        }
    }

    #[getter]
    fn site_buses(&self) -> Vec<PySiteBus> {
        self.inner
            .site_buses
            .iter()
            .map(|b| PySiteBus { inner: b.clone() })
            .collect()
    }

    #[getter]
    fn word_buses(&self) -> Vec<PyWordBus> {
        self.inner
            .word_buses
            .iter()
            .map(|b| PyWordBus { inner: b.clone() })
            .collect()
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
    fn entangling_pairs(&self) -> Vec<(u32, u32)> {
        self.inner
            .entangling_pairs
            .iter()
            .map(|p| (p[0], p[1]))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Zone(grid={:?}, site_buses={}, word_buses={})",
            self.inner.grid,
            self.inner.site_buses.len(),
            self.inner.word_buses.len()
        )
    }
}

// ── Mode ──

#[pyclass(name = "Mode", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyMode {
    pub(crate) inner: rs::Mode,
}

#[pymethods]
impl PyMode {
    #[new]
    fn new(
        name: String,
        zones: Vec<i64>,
        bitstring_order: Vec<PyRef<'_, PyLocationAddr>>,
    ) -> PyResult<Self> {
        let zones = validate_vec::<u32>("zones", zones)?;
        Ok(Self {
            inner: rs::Mode {
                name,
                zones,
                bitstring_order: bitstring_order.iter().map(|l| l.inner).collect(),
            },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn zones(&self) -> Vec<u32> {
        self.inner.zones.clone()
    }

    #[getter]
    fn bitstring_order(&self) -> Vec<PyLocationAddr> {
        self.inner
            .bitstring_order
            .iter()
            .map(|l| PyLocationAddr { inner: *l })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Mode(name='{}', zones={:?})",
            self.inner.name, self.inner.zones
        )
    }
}

// ── TransportPath ──

#[pyclass(
    name = "TransportPath",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyTransportPath {
    pub(crate) inner: rs::TransportPath,
}

#[pymethods]
impl PyTransportPath {
    #[new]
    fn new(lane: &PyLaneAddr, waypoints: Vec<(f64, f64)>) -> PyResult<Self> {
        let path = rs::TransportPath {
            lane: lane.inner.encode_u64(),
            waypoints: waypoints.into_iter().map(|(x, y)| [x, y]).collect(),
        };
        if !path.check_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "waypoints contain non-finite coordinate (NaN or Inf)",
            ));
        }
        Ok(Self { inner: path })
    }

    #[getter]
    fn lane(&self) -> PyLaneAddr {
        PyLaneAddr {
            inner: rs_addr::LaneAddr::decode_u64(self.inner.lane),
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
            "TransportPath(lane=0x{:016X}, waypoints={})",
            self.inner.lane,
            self.inner.waypoints.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ── ArchSpec ──

#[pyclass(name = "ArchSpec", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyArchSpec {
    pub(crate) inner: rs::ArchSpec,
}

#[pymethods]
impl PyArchSpec {
    #[new]
    #[pyo3(signature = (version, words, zones, zone_buses, modes, paths=None, feed_forward=false, atom_reloading=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        version: (u16, u16),
        words: Vec<PyRef<'_, PyWord>>,
        zones: Vec<PyRef<'_, PyZone>>,
        zone_buses: Vec<PyRef<'_, PyZoneBus>>,
        modes: Vec<PyRef<'_, PyMode>>,
        paths: Option<Vec<PyRef<'_, PyTransportPath>>>,
        feed_forward: bool,
        atom_reloading: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: rs::ArchSpec {
                version: Version::new(version.0, version.1),
                words: words.iter().map(|w| w.inner.clone()).collect(),
                zones: zones.iter().map(|z| z.inner.clone()).collect(),
                zone_buses: zone_buses.iter().map(|b| b.inner.clone()).collect(),
                modes: modes.iter().map(|m| m.inner.clone()).collect(),
                paths: paths.map(|v| v.iter().map(|p| p.inner.clone()).collect()),
                feed_forward,
                atom_reloading,
            },
        })
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
    fn words(&self) -> Vec<PyWord> {
        self.inner
            .words
            .iter()
            .map(|w| PyWord { inner: w.clone() })
            .collect()
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
    fn zone_buses(&self) -> Vec<PyZoneBus> {
        self.inner
            .zone_buses
            .iter()
            .map(|b| PyZoneBus { inner: b.clone() })
            .collect()
    }

    #[getter]
    fn modes(&self) -> Vec<PyMode> {
        self.inner
            .modes
            .iter()
            .map(|m| PyMode { inner: m.clone() })
            .collect()
    }

    #[getter]
    fn sites_per_word(&self) -> usize {
        self.inner.sites_per_word()
    }

    #[getter]
    fn feed_forward(&self) -> bool {
        self.inner.feed_forward
    }

    #[getter]
    fn atom_reloading(&self) -> bool {
        self.inner.atom_reloading
    }

    #[getter]
    fn paths(&self) -> Option<Vec<PyTransportPath>> {
        self.inner.paths.as_ref().map(|v| {
            v.iter()
                .map(|p| PyTransportPath { inner: p.clone() })
                .collect()
        })
    }

    fn word_by_id(&self, id: i64) -> PyResult<Option<PyWord>> {
        let id = validate_field::<u32>("id", id)?;
        Ok(self
            .inner
            .word_by_id(id)
            .map(|w| PyWord { inner: w.clone() }))
    }

    fn zone_by_id(&self, id: i64) -> PyResult<Option<PyZone>> {
        let id = validate_field::<u32>("id", id)?;
        Ok(self
            .inner
            .zone_by_id(id)
            .map(|z| PyZone { inner: z.clone() }))
    }

    /// Resolve a location address to its physical (x, y) coordinates.
    ///
    /// Returns None if the zone, word, or site is not found.
    #[pyo3(text_signature = "(self, loc)")]
    fn location_position(&self, loc: &PyLocationAddr) -> Option<(f64, f64)> {
        self.inner.location_position(&loc.inner)
    }

    /// Resolve a lane address to its source and destination location addresses.
    ///
    /// Given a ``LaneAddr``, determines which two ``LocationAddr`` endpoints the
    /// lane connects by tracing through the appropriate bus (site bus, word bus,
    /// or zone bus) in the specified direction (forward or backward).
    ///
    /// Returns a ``(src, dst)`` tuple of ``LocationAddr``, or None if the lane
    /// references an invalid bus, word, or site.
    #[pyo3(text_signature = "(self, lane)")]
    fn lane_endpoints(&self, lane: &PyLaneAddr) -> Option<(PyLocationAddr, PyLocationAddr)> {
        let (src, dst) = self.inner.lane_endpoints(&lane.inner)?;
        Some((PyLocationAddr { inner: src }, PyLocationAddr { inner: dst }))
    }

    /// Get the CZ partner for a given location.
    ///
    /// For a site in a zone, finds the partner word from the zone's
    /// ``entangling_pairs`` and returns the corresponding location.
    /// Returns None if the word is not in any entangling pair.
    #[pyo3(text_signature = "(self, loc)")]
    fn get_cz_partner(&self, loc: &PyLocationAddr) -> Option<PyLocationAddr> {
        self.inner
            .get_cz_partner(&loc.inner)
            .map(|l| PyLocationAddr { inner: l })
    }

    // -- Derived topology queries (#464 phase 2) --

    /// Build a bidirectional word-partner map from entangling pairs.
    ///
    /// Returns a dict mapping each word_id to its CZ partner word_id.
    fn word_partner_map(&self) -> std::collections::HashMap<u32, u32> {
        self.inner.word_partner_map()
    }

    /// Map each word_id to the zone_id that owns it.
    ///
    /// Returns a dict mapping word_id → zone_id.
    fn word_zone_map(&self) -> std::collections::HashMap<u32, u32> {
        self.inner.word_zone_map()
    }

    /// Return sorted left-CZ word IDs (lower word of each CZ pair + unpaired).
    fn left_cz_word_ids(&self) -> Vec<u32> {
        self.inner.left_cz_word_ids()
    }

    /// Reverse-lookup: find the lane connecting src → dst.
    ///
    /// Returns the ``LaneAddress`` if found, or None.
    #[pyo3(text_signature = "(self, src, dst)")]
    fn lane_for_endpoints(&self, src: &PyLocationAddr, dst: &PyLocationAddr) -> Option<PyLaneAddr> {
        self.inner
            .lane_for_endpoints(&src.inner, &dst.inner)
            .map(|l| PyLaneAddr { inner: l })
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

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
