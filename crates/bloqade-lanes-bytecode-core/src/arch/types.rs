use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::version::Version;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchSpec {
    pub version: Version,
    pub geometry: Geometry,
    pub buses: Buses,
    pub words_with_site_buses: Vec<u32>,
    pub sites_with_word_buses: Vec<u32>,
    pub zones: Vec<Zone>,
    pub entangling_zones: Vec<u32>,
    pub measurement_mode_zones: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paths: Option<Vec<TransportPath>>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geometry {
    pub sites_per_word: u32,
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub grid: Grid,
    /// Each entry is `[x_idx, y_idx]` indexing into the grid's x and y
    /// coordinate arrays.
    pub sites: Vec<[u32; 2]>,
    /// Optional. `cz_pairs[i]` is `[word_id, site_id]` — the site that
    /// site `i` entangles with during CZ.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cz_pairs: Option<Vec<[u32; 2]>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grid {
    pub x_start: f64,
    pub y_start: f64,
    pub x_spacing: Vec<f64>,
    pub y_spacing: Vec<f64>,
}

impl Grid {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Buses {
    pub site_buses: Vec<Bus>,
    pub word_buses: Vec<Bus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bus {
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub words: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::example_arch_spec;

    #[test]
    fn serde_round_trip() {
        let spec = example_arch_spec();
        let json = serde_json::to_string(&spec).unwrap();
        let deserialized: ArchSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, deserialized);
    }

    #[test]
    fn optional_fields_absent() {
        let json = r#"{
            "version": 1,
            "geometry": {
                "sites_per_word": 2,
                "words": [
                    {
                        "grid": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [], "y_spacing": [2.0] },
                        "sites": [[0, 0], [0, 1]]
                    }
                ]
            },
            "buses": { "site_buses": [], "word_buses": [] },
            "words_with_site_buses": [],
            "sites_with_word_buses": [],
            "zones": [{ "words": [0] }],
            "entangling_zones": [],
            "measurement_mode_zones": [0]
        }"#;
        let spec: ArchSpec = serde_json::from_str(json).unwrap();
        assert!(spec.paths.is_none());
        assert!(spec.geometry.words[0].cz_pairs.is_none());
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
}
