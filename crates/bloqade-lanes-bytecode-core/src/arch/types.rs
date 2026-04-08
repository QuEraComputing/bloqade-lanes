use std::hash::{Hash, Hasher};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::addr::{LocationAddr, SiteRef, WordRef, ZonedWordRef};
use crate::version::Version;

/// Normalize -0.0 to 0.0 for consistent hashing (since -0.0 == 0.0
/// under PartialEq but to_bits() differs).
#[inline]
fn canonical_f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0_f64.to_bits()
    } else {
        v.to_bits()
    }
}

/// A transport bus that maps source positions to destination positions.
///
/// The `src` and `dst` lists are parallel arrays: `src[i]` maps to `dst[i]`.
/// The type parameter `T` determines the address type used for bus entries:
/// - `Bus<SiteRef>` — site buses within a zone
/// - `Bus<WordRef>` — word buses within a zone
/// - `Bus<ZonedWordRef>` — inter-zone word buses
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "T: Serialize + for<'de2> Deserialize<'de2>")]
pub struct Bus<T> {
    /// Source indices.
    pub src: Vec<T>,
    /// Destination indices (same length as `src`).
    pub dst: Vec<T>,
}

/// A group of atom sites that share a coordinate grid.
///
/// Each word contains a fixed number of sites. Sites are positioned on the
/// parent zone's grid via `[x_idx, y_idx]` index pairs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Word {
    /// Each entry is `[x_idx, y_idx]` indexing into the parent zone's grid
    /// x and y coordinate arrays.
    pub sites: Vec<[u32; 2]>,
}

/// A logical zone grouping words with a shared coordinate grid and buses.
///
/// Each zone owns its grid and the site/word buses that operate within it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    /// Coordinate grid for all words in this zone.
    pub grid: Grid,
    /// Site buses that move atoms between sites within words of this zone.
    pub site_buses: Vec<Bus<SiteRef>>,
    /// Word buses that move atoms between words within this zone.
    pub word_buses: Vec<Bus<WordRef>>,
    /// Word IDs (within this zone) that have site-bus transport capability.
    pub words_with_site_buses: Vec<u32>,
    /// Site indices that participate in word-bus transport within this zone.
    pub sites_with_word_buses: Vec<u32>,
    /// Word pairs within this zone that are at blockade radius for CZ gates.
    /// A zone with empty `entangling_pairs` is a storage/low-connectivity zone.
    #[serde(default)]
    pub entangling_pairs: Vec<[u32; 2]>,
}

/// A named operational mode for the device.
///
/// Modes define subsets of zones and the bitstring ordering used for
/// measurement results.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mode {
    /// Human-readable mode name.
    pub name: String,
    /// Zone IDs active in this mode.
    pub zones: Vec<u32>,
    /// Bit-to-location mapping for measurement results.
    pub bitstring_order: Vec<LocationAddr>,
}

/// Architecture specification for a quantum device.
///
/// Describes the full hardware topology: words, zones (each owning a grid
/// and intra-zone buses), inter-zone buses, entangling pairs, operational
/// modes, and device capabilities. Can be loaded from JSON via
/// [`from_json`](ArchSpec::from_json) /
/// [`from_json_validated`](ArchSpec::from_json_validated),
/// or constructed programmatically.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArchSpec {
    /// Spec format version.
    pub version: Version,
    /// Word definitions. A word's ID is its index in this list.
    pub words: Vec<Word>,
    /// Logical zones, each owning a grid and intra-zone buses.
    pub zones: Vec<Zone>,
    /// Inter-zone word buses.
    pub zone_buses: Vec<Bus<ZonedWordRef>>,
    /// Operational modes (measurement, etc.).
    pub modes: Vec<Mode>,
    /// Optional AOD transport paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paths: Option<Vec<TransportPath>>,
    /// Whether the device supports mid-circuit measurement with classical feedback.
    /// Defaults to `false` when absent in JSON.
    #[serde(default)]
    pub feed_forward: bool,
    /// Whether the device supports reloading atoms after initial fill.
    /// Defaults to `false` when absent in JSON.
    #[serde(default)]
    pub atom_reloading: bool,
}

impl ArchSpec {
    /// Number of sites in each word. Returns 0 if there are no words.
    pub fn sites_per_word(&self) -> usize {
        self.words.first().map_or(0, |w| w.sites.len())
    }

    /// Construct an ArchSpec from all components, validating on creation.
    ///
    /// Returns the validated ArchSpec, or a list of validation errors.
    /// This is the primary construction path — prefer this over building
    /// the struct directly to ensure invariants hold.
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        version: Version,
        words: Vec<Word>,
        zones: Vec<Zone>,
        zone_buses: Vec<Bus<ZonedWordRef>>,
        modes: Vec<Mode>,
        paths: Option<Vec<TransportPath>>,
        feed_forward: bool,
        atom_reloading: bool,
    ) -> Result<Self, Vec<super::validate::ArchSpecError>> {
        let spec = Self {
            version,
            words,
            zones,
            zone_buses,
            modes,
            paths,
            feed_forward,
            atom_reloading,
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// A transport path for a lane, defined as a sequence of (x, y) waypoints.
/// The lane is identified by its encoded `LaneAddr` (serialized as a hex string).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportPath {
    /// Encoded `LaneAddr` identifying the transport lane.
    /// Serialized as a `"0x..."` hex string in JSON.
    #[serde(
        serialize_with = "serialize_lane_hex",
        deserialize_with = "deserialize_lane_hex"
    )]
    pub lane: u64,
    /// Sequence of `[x, y]` waypoints defining the physical trajectory.
    pub waypoints: Vec<[f64; 2]>,
}

impl TransportPath {
    /// Check that all waypoint coordinates are finite (not NaN or Inf).
    pub fn check_finite(&self) -> bool {
        self.waypoints
            .iter()
            .all(|wp| wp[0].is_finite() && wp[1].is_finite())
    }
}

impl Eq for TransportPath {}

impl Hash for TransportPath {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.lane.hash(state);
        for wp in &self.waypoints {
            canonical_f64_bits(wp[0]).hash(state);
            canonical_f64_bits(wp[1]).hash(state);
        }
    }
}

fn serialize_lane_hex<S: Serializer>(lane: &u64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("0x{:016X}", lane))
}

fn deserialize_lane_hex<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    let s = String::deserialize(deserializer)?;
    let hex = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .ok_or_else(|| {
            serde::de::Error::custom(format!(
                "expected hex string starting with '0x', got '{}'",
                s
            ))
        })?;
    if hex.len() != 16 {
        return Err(serde::de::Error::custom(format!(
            "expected exactly 16 hex digits after '0x', got {} in '{}'",
            hex.len(),
            s
        )));
    }
    u64::from_str_radix(hex, 16).map_err(serde::de::Error::custom)
}

/// A 2D coordinate grid for positioning atom sites within a word.
///
/// Positions are computed as cumulative sums from the start value:
/// `x[i] = x_start + sum(x_spacing[0..i])`. The number of grid points
/// along each axis is `len(spacing) + 1`.
///
/// All coordinate values must be finite (no NaN or Inf).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    /// X-coordinate of the first grid point.
    pub x_start: f64,
    /// Y-coordinate of the first grid point.
    pub y_start: f64,
    /// Spacing between consecutive x-coordinates.
    pub x_spacing: Vec<f64>,
    /// Spacing between consecutive y-coordinates.
    pub y_spacing: Vec<f64>,
}

impl Eq for Grid {}

impl Hash for Grid {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonical_f64_bits(self.x_start).hash(state);
        canonical_f64_bits(self.y_start).hash(state);
        for v in &self.x_spacing {
            canonical_f64_bits(*v).hash(state);
        }
        for v in &self.y_spacing {
            canonical_f64_bits(*v).hash(state);
        }
    }
}

impl Grid {
    /// Check that all float values are finite (not NaN or Inf).
    pub fn check_finite(&self) -> Result<(), &'static str> {
        if !self.x_start.is_finite() {
            return Err("x_start");
        }
        if !self.y_start.is_finite() {
            return Err("y_start");
        }
        if let Some(v) = self.x_spacing.iter().find(|v| !v.is_finite()) {
            let _ = v;
            return Err("x_spacing");
        }
        if let Some(v) = self.y_spacing.iter().find(|v| !v.is_finite()) {
            let _ = v;
            return Err("y_spacing");
        }
        Ok(())
    }

    /// Construct a `Grid` from explicit position arrays.
    ///
    /// The first element becomes the start value and consecutive differences
    /// become the spacing vector.  Panics if either slice is empty.
    pub fn from_positions(x_positions: &[f64], y_positions: &[f64]) -> Self {
        assert!(
            !x_positions.is_empty(),
            "x_positions must have at least one element"
        );
        assert!(
            !y_positions.is_empty(),
            "y_positions must have at least one element"
        );
        Self {
            x_start: x_positions[0],
            y_start: y_positions[0],
            x_spacing: x_positions.windows(2).map(|w| w[1] - w[0]).collect(),
            y_spacing: y_positions.windows(2).map(|w| w[1] - w[0]).collect(),
        }
    }

    /// Number of x-axis grid points.
    pub fn num_x(&self) -> usize {
        self.x_spacing.len() + 1
    }

    /// Number of y-axis grid points.
    pub fn num_y(&self) -> usize {
        self.y_spacing.len() + 1
    }

    /// Compute the x-coordinate at the given index.
    pub fn x_position(&self, idx: usize) -> Option<f64> {
        if idx >= self.num_x() {
            return None;
        }
        Some(self.x_start + self.x_spacing[..idx].iter().sum::<f64>())
    }

    /// Compute the y-coordinate at the given index.
    pub fn y_position(&self, idx: usize) -> Option<f64> {
        if idx >= self.num_y() {
            return None;
        }
        Some(self.y_start + self.y_spacing[..idx].iter().sum::<f64>())
    }

    /// Compute all x-coordinates as a Vec.
    pub fn x_positions(&self) -> Vec<f64> {
        let mut positions = Vec::with_capacity(self.num_x());
        let mut acc = self.x_start;
        positions.push(acc);
        for &dx in &self.x_spacing {
            acc += dx;
            positions.push(acc);
        }
        positions
    }

    /// Compute all y-coordinates as a Vec.
    pub fn y_positions(&self) -> Vec<f64> {
        let mut positions = Vec::with_capacity(self.num_y());
        let mut acc = self.y_start;
        positions.push(acc);
        for &dy in &self.y_spacing {
            acc += dy;
            positions.push(acc);
        }
        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_site_bus_serde() {
        let bus: Bus<SiteRef> = Bus {
            src: vec![SiteRef(0), SiteRef(1)],
            dst: vec![SiteRef(3), SiteRef(4)],
        };
        let json = serde_json::to_string(&bus).unwrap();
        let deserialized: Bus<SiteRef> = serde_json::from_str(&json).unwrap();
        assert_eq!(bus.src, deserialized.src);
        assert_eq!(bus.dst, deserialized.dst);
    }

    #[test]
    fn test_zone_bus_serde() {
        let bus: Bus<ZonedWordRef> = Bus {
            src: vec![ZonedWordRef {
                zone_id: 0,
                word_id: 1,
            }],
            dst: vec![ZonedWordRef {
                zone_id: 1,
                word_id: 1,
            }],
        };
        let json = serde_json::to_string(&bus).unwrap();
        let deserialized: Bus<ZonedWordRef> = serde_json::from_str(&json).unwrap();
        assert_eq!(bus.src, deserialized.src);
    }

    #[test]
    fn lane_hex_canonical_accepted() {
        let json = r#"{"lane": "0x0000000000000001", "waypoints": [[1.0, 2.0]]}"#;
        let path: TransportPath = serde_json::from_str(json).unwrap();
        assert_eq!(path.lane, 1);
    }

    #[test]
    fn lane_hex_too_short_rejected() {
        let json = r#"{"lane": "0x1", "waypoints": []}"#;
        let err = serde_json::from_str::<TransportPath>(json).unwrap_err();
        assert!(err.to_string().contains("expected exactly 16 hex digits"));
    }

    #[test]
    fn lane_hex_too_long_rejected() {
        let json = r#"{"lane": "0x12345678901234567", "waypoints": []}"#;
        let err = serde_json::from_str::<TransportPath>(json).unwrap_err();
        assert!(err.to_string().contains("expected exactly 16 hex digits"));
    }

    #[test]
    fn grid_from_positions_uniform() {
        let grid = Grid::from_positions(&[1.0, 3.0, 5.0], &[2.0, 4.5]);
        assert_eq!(grid.x_start, 1.0);
        assert_eq!(grid.y_start, 2.0);
        assert_eq!(grid.x_spacing, vec![2.0, 2.0]);
        assert_eq!(grid.y_spacing, vec![2.5]);
        assert_eq!(grid.num_x(), 3);
        assert_eq!(grid.num_y(), 2);
        assert_eq!(grid.x_positions(), vec![1.0, 3.0, 5.0]);
        assert_eq!(grid.y_positions(), vec![2.0, 4.5]);
    }

    #[test]
    fn grid_from_positions_single() {
        let grid = Grid::from_positions(&[7.0], &[3.0]);
        assert_eq!(grid.x_start, 7.0);
        assert_eq!(grid.y_start, 3.0);
        assert!(grid.x_spacing.is_empty());
        assert!(grid.y_spacing.is_empty());
        assert_eq!(grid.num_x(), 1);
        assert_eq!(grid.num_y(), 1);
    }

    #[test]
    #[should_panic(expected = "x_positions must have at least one element")]
    fn grid_from_positions_empty_x() {
        Grid::from_positions(&[], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "y_positions must have at least one element")]
    fn grid_from_positions_empty_y() {
        Grid::from_positions(&[1.0], &[]);
    }

    #[test]
    fn lane_hex_missing_prefix_rejected() {
        let json = r#"{"lane": "00000001", "waypoints": []}"#;
        let err = serde_json::from_str::<TransportPath>(json).unwrap_err();
        assert!(
            err.to_string()
                .contains("expected hex string starting with '0x'")
        );
    }

    #[test]
    fn test_sites_per_word() {
        let spec = ArchSpec {
            version: Version::new(2, 0),
            words: vec![
                Word {
                    sites: vec![[0, 0], [1, 0]],
                },
                Word {
                    sites: vec![[0, 0], [1, 0]],
                },
            ],
            zones: vec![],
            zone_buses: vec![],
            modes: vec![],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
        };
        assert_eq!(spec.sites_per_word(), 2);
    }

    #[test]
    fn test_sites_per_word_empty() {
        let spec = ArchSpec {
            version: Version::new(2, 0),
            words: vec![],
            zones: vec![],
            zone_buses: vec![],
            modes: vec![],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
        };
        assert_eq!(spec.sites_per_word(), 0);
    }
}
