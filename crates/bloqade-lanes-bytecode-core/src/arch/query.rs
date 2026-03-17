use std::collections::HashSet;
use std::fmt;

use thiserror::Error;

use super::addr::{LaneAddr, LocationAddr, MoveType};
use super::types::{ArchSpec, Bus, Word};
use super::validate::ArchSpecError;

#[derive(Debug, Error)]
pub enum ArchSpecLoadError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("validation errors: {0:?}")]
    Validation(Vec<ArchSpecError>),
}

impl From<Vec<ArchSpecError>> for ArchSpecLoadError {
    fn from(errors: Vec<ArchSpecError>) -> Self {
        ArchSpecLoadError::Validation(errors)
    }
}

// --- Group-level error types ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocationGroupError {
    /// A location address appears more than once in the group.
    DuplicateAddress { address: u32 },
    /// A location address is invalid per the arch spec.
    InvalidAddress { word_id: u32, site_id: u32 },
}

impl fmt::Display for LocationGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocationGroupError::DuplicateAddress { address } => {
                let addr = LocationAddr::decode(*address);
                write!(
                    f,
                    "duplicate location address word_id={}, site_id={}",
                    addr.word_id, addr.site_id
                )
            }
            LocationGroupError::InvalidAddress { word_id, site_id } => {
                write!(
                    f,
                    "invalid location word_id={}, site_id={}",
                    word_id, site_id
                )
            }
        }
    }
}

impl std::error::Error for LocationGroupError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneGroupError {
    /// A lane address appears more than once in the group.
    DuplicateAddress { address: (u32, u32) },
    /// A lane address is invalid per the arch spec.
    InvalidLane { message: String },
    /// Lanes have inconsistent bus_id, move_type, or direction.
    Inconsistent { message: String },
    /// Lane word_id not in words_with_site_buses.
    WordNotInSiteBusList { word_id: u32 },
    /// Lane site_id not in sites_with_word_buses.
    SiteNotInWordBusList { site_id: u32 },
    /// Lane group violates AOD constraint (e.g. not a complete rectangle).
    AODConstraintViolation { message: String },
}

impl fmt::Display for LaneGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaneGroupError::DuplicateAddress { address } => {
                let combined = (address.0 as u64) | ((address.1 as u64) << 32);
                write!(f, "duplicate lane address 0x{:016x}", combined)
            }
            LaneGroupError::InvalidLane { message } => {
                write!(f, "invalid lane: {}", message)
            }
            LaneGroupError::Inconsistent { message } => {
                write!(f, "lane group inconsistent: {}", message)
            }
            LaneGroupError::WordNotInSiteBusList { word_id } => {
                write!(f, "word_id {} not in words_with_site_buses", word_id)
            }
            LaneGroupError::SiteNotInWordBusList { site_id } => {
                write!(f, "site_id {} not in sites_with_word_buses", site_id)
            }
            LaneGroupError::AODConstraintViolation { message } => {
                write!(f, "AOD constraint violation: {}", message)
            }
        }
    }
}

impl std::error::Error for LaneGroupError {}

// --- Bus methods ---

impl Bus {
    /// Given a source value, return the destination value (forward move).
    /// For site buses, this maps source site → destination site.
    /// For word buses, this maps source word → destination word.
    pub fn resolve_forward(&self, src: u32) -> Option<u32> {
        self.src.iter().position(|&s| s == src).map(|i| self.dst[i])
    }

    /// Given a destination value, return the source value (backward move).
    /// For site buses, this maps destination site → source site.
    /// For word buses, this maps destination word → source word.
    pub fn resolve_backward(&self, dst: u32) -> Option<u32> {
        self.dst.iter().position(|&d| d == dst).map(|i| self.src[i])
    }
}

// --- Word methods ---

impl Word {
    /// Resolve a site's grid indices to physical (x, y) coordinates.
    pub fn site_position(&self, site_idx: usize) -> Option<(f64, f64)> {
        let pair = self.site_indices.get(site_idx)?;
        let x = self.positions.x_position(pair[0] as usize)?;
        let y = self.positions.y_position(pair[1] as usize)?;
        Some((x, y))
    }
}

// --- ArchSpec methods ---

impl ArchSpec {
    // -- Deserialization --

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Deserialize from JSON and validate.
    pub fn from_json_validated(json: &str) -> Result<Self, ArchSpecLoadError> {
        let spec = Self::from_json(json)?;
        spec.validate()?;
        Ok(spec)
    }

    // -- Lookup helpers --

    pub fn word_by_id(&self, id: u32) -> Option<&Word> {
        self.geometry.words.get(id as usize)
    }

    pub fn zone_by_id(&self, id: u32) -> Option<&super::types::Zone> {
        self.zones.get(id as usize)
    }

    pub fn site_bus_by_id(&self, id: u32) -> Option<&Bus> {
        self.buses.site_buses.get(id as usize)
    }

    pub fn word_bus_by_id(&self, id: u32) -> Option<&Bus> {
        self.buses.word_buses.get(id as usize)
    }

    // -- Position resolution --

    /// Resolve a LocationAddr to physical (x, y) coordinates.
    pub fn location_position(&self, loc: &crate::arch::addr::LocationAddr) -> Option<(f64, f64)> {
        let word = self.word_by_id(loc.word_id)?;
        word.site_position(loc.site_id as usize)
    }

    /// Resolve a `LaneAddr` to its source and destination `LocationAddr` pair.
    ///
    /// Returns `Some((src, dst))` if the lane can be resolved through the bus,
    /// or `None` if the lane references invalid words, sites, or buses.
    pub fn lane_endpoints(
        &self,
        lane: &crate::arch::addr::LaneAddr,
    ) -> Option<(LocationAddr, LocationAddr)> {
        use crate::arch::addr::{Direction, MoveType};

        let src = LocationAddr {
            word_id: lane.word_id,
            site_id: lane.site_id,
        };

        let dst = match (lane.move_type, lane.direction) {
            (MoveType::SiteBus, Direction::Forward) => {
                let bus = self.site_bus_by_id(lane.bus_id)?;
                let dst_site = bus.resolve_forward(lane.site_id)?;
                LocationAddr {
                    word_id: lane.word_id,
                    site_id: dst_site,
                }
            }
            (MoveType::SiteBus, Direction::Backward) => {
                let bus = self.site_bus_by_id(lane.bus_id)?;
                let dst_site = bus.resolve_backward(lane.site_id)?;
                LocationAddr {
                    word_id: lane.word_id,
                    site_id: dst_site,
                }
            }
            (MoveType::WordBus, Direction::Forward) => {
                let bus = self.word_bus_by_id(lane.bus_id)?;
                let dst_word = bus.resolve_forward(lane.word_id)?;
                LocationAddr {
                    word_id: dst_word,
                    site_id: lane.site_id,
                }
            }
            (MoveType::WordBus, Direction::Backward) => {
                let bus = self.word_bus_by_id(lane.bus_id)?;
                let dst_word = bus.resolve_backward(lane.word_id)?;
                LocationAddr {
                    word_id: dst_word,
                    site_id: lane.site_id,
                }
            }
        };

        Some((src, dst))
    }

    // -- Address validation --

    /// Check whether a location address (word_id, site_id) is valid.
    pub fn check_location(&self, loc: &crate::arch::addr::LocationAddr) -> Option<String> {
        let num_words = self.geometry.words.len() as u32;
        let sites_per_word = self.geometry.sites_per_word;
        if loc.word_id >= num_words || loc.site_id >= sites_per_word {
            Some(format!(
                "invalid location word_id={}, site_id={}",
                loc.word_id, loc.site_id
            ))
        } else {
            None
        }
    }

    /// Check whether a lane address is valid (bus exists, word/site in range).
    pub fn check_lane(&self, addr: &LaneAddr) -> Vec<String> {
        let num_words = self.geometry.words.len() as u32;
        let sites_per_word = self.geometry.sites_per_word;
        let mut errors = Vec::new();

        match addr.move_type {
            MoveType::SiteBus => {
                if self.site_bus_by_id(addr.bus_id).is_none() {
                    errors.push(format!("unknown site_bus id {}", addr.bus_id));
                }
                if addr.word_id >= num_words {
                    errors.push(format!("word_id {} out of range", addr.word_id));
                }
                if addr.site_id >= sites_per_word {
                    errors.push(format!("site_id {} out of range", addr.site_id));
                }
            }
            MoveType::WordBus => {
                if self.word_bus_by_id(addr.bus_id).is_none() {
                    errors.push(format!("unknown word_bus id {}", addr.bus_id));
                }
                if addr.word_id >= num_words {
                    errors.push(format!("word_id {} out of range", addr.word_id));
                }
            }
        }
        errors
    }

    /// Check whether a zone address is valid.
    pub fn check_zone(&self, zone: &crate::arch::addr::ZoneAddr) -> Option<String> {
        if self.zone_by_id(zone.zone_id).is_none() {
            Some(format!("invalid zone_id={}", zone.zone_id))
        } else {
            None
        }
    }

    // -- Group validation --

    /// Check that a group of lanes share consistent bus_id, move_type, and direction.
    pub fn check_lane_group_consistency(&self, lanes: &[LaneAddr]) -> Vec<String> {
        if lanes.is_empty() {
            return vec![];
        }
        let first = &lanes[0];
        let mut errors = Vec::new();

        for lane in &lanes[1..] {
            if lane.bus_id != first.bus_id {
                errors.push(format!(
                    "bus_id mismatch: expected {}, got {}",
                    first.bus_id, lane.bus_id
                ));
            }
            if lane.move_type != first.move_type {
                errors.push(format!(
                    "move_type mismatch: expected {:?}, got {:?}",
                    first.move_type, lane.move_type
                ));
            }
            if lane.direction != first.direction {
                errors.push(format!(
                    "direction mismatch: expected {:?}, got {:?}",
                    first.direction, lane.direction
                ));
            }
        }

        errors
    }

    /// Check that each lane's word/site belongs to the correct bus membership list.
    /// Returns unique `(word_ids_not_in_site_bus_list, site_ids_not_in_word_bus_list)`.
    pub fn check_lane_group_membership(&self, lanes: &[LaneAddr]) -> (Vec<u32>, Vec<u32>) {
        use std::collections::BTreeSet;

        let mut bad_words = BTreeSet::new();
        let mut bad_sites = BTreeSet::new();

        for lane in lanes {
            match lane.move_type {
                MoveType::SiteBus => {
                    if !self.words_with_site_buses.contains(&lane.word_id) {
                        bad_words.insert(lane.word_id);
                    }
                }
                MoveType::WordBus => {
                    if !self.sites_with_word_buses.contains(&lane.site_id) {
                        bad_sites.insert(lane.site_id);
                    }
                }
            }
        }

        (
            bad_words.into_iter().collect(),
            bad_sites.into_iter().collect(),
        )
    }

    /// Validate a group of location addresses: checks each address against the
    /// arch spec and checks for duplicates within the group.
    pub fn check_locations(&self, locations: &[LocationAddr]) -> Vec<LocationGroupError> {
        let mut errors = Vec::new();

        // Check each unique address is valid (report once per unique address)
        let mut checked = HashSet::new();
        for loc in locations {
            let bits = loc.encode();
            if checked.insert(bits) && self.check_location(loc).is_some() {
                errors.push(LocationGroupError::InvalidAddress {
                    word_id: loc.word_id,
                    site_id: loc.site_id,
                });
            }
        }

        // Check for duplicates (report once per unique duplicated address)
        let mut seen = HashSet::new();
        let mut reported = HashSet::new();
        for loc in locations {
            let bits = loc.encode();
            if !seen.insert(bits) && reported.insert(bits) {
                errors.push(LocationGroupError::DuplicateAddress { address: bits });
            }
        }

        errors
    }

    /// Validate a group of lane addresses: checks each address against the
    /// arch spec, checks for duplicates, and (when more than one lane)
    /// validates consistency, bus membership, and AOD constraints.
    pub fn check_lanes(&self, lanes: &[LaneAddr]) -> Vec<LaneGroupError> {
        let mut errors = Vec::new();

        // Check each unique address is valid (report once per unique address)
        let mut checked = HashSet::new();
        for lane in lanes {
            let bits = lane.encode();
            if checked.insert(bits) {
                for msg in self.check_lane(lane) {
                    errors.push(LaneGroupError::InvalidLane { message: msg });
                }
            }
        }

        // Check for duplicates (report once per unique duplicated address)
        let mut seen = HashSet::new();
        let mut reported = HashSet::new();
        for lane in lanes {
            let pair = lane.encode();
            if !seen.insert(pair) && reported.insert(pair) {
                errors.push(LaneGroupError::DuplicateAddress { address: pair });
            }
        }

        // Group-level checks (only meaningful with >1 lane)
        if lanes.len() > 1 {
            for msg in self.check_lane_group_consistency(lanes) {
                errors.push(LaneGroupError::Inconsistent { message: msg });
            }
            let (bad_words, bad_sites) = self.check_lane_group_membership(lanes);
            for word_id in bad_words {
                errors.push(LaneGroupError::WordNotInSiteBusList { word_id });
            }
            for site_id in bad_sites {
                errors.push(LaneGroupError::SiteNotInWordBusList { site_id });
            }
            for msg in self.check_lane_group_geometry(lanes) {
                errors.push(LaneGroupError::AODConstraintViolation { message: msg });
            }
        }

        errors
    }

    /// Check AOD constraint: lane positions must form a complete rectangle
    /// (Cartesian product of unique X and Y values).
    pub fn check_lane_group_geometry(&self, lanes: &[LaneAddr]) -> Vec<String> {
        use std::collections::BTreeSet;

        let positions: Vec<(f64, f64)> = lanes
            .iter()
            .filter_map(|lane| {
                let loc = crate::arch::addr::LocationAddr {
                    word_id: lane.word_id,
                    site_id: lane.site_id,
                };
                self.location_position(&loc)
            })
            .collect();

        if positions.len() != lanes.len() {
            return vec!["some lane positions could not be resolved".to_string()];
        }

        let unique_x: BTreeSet<u64> = positions.iter().map(|(x, _)| x.to_bits()).collect();
        let unique_y: BTreeSet<u64> = positions.iter().map(|(_, y)| y.to_bits()).collect();

        let expected: BTreeSet<(u64, u64)> = unique_x
            .iter()
            .flat_map(|x| unique_y.iter().map(move |y| (*x, *y)))
            .collect();

        let actual: BTreeSet<(u64, u64)> = positions
            .iter()
            .map(|(x, y)| (x.to_bits(), y.to_bits()))
            .collect();

        if actual != expected {
            vec![format!(
                "lanes do not form a complete rectangle: expected {} positions ({}x × {}y), got {} unique positions",
                expected.len(),
                unique_x.len(),
                unique_y.len(),
                actual.len()
            )]
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::arch::example_arch_spec;

    #[test]
    fn word_by_id_found() {
        let spec = example_arch_spec();
        let word = spec.word_by_id(0).unwrap();
        assert_eq!(word.site_indices.len(), 10);
    }

    #[test]
    fn word_by_id_not_found() {
        let spec = example_arch_spec();
        assert!(spec.word_by_id(99).is_none());
    }

    #[test]
    fn zone_by_id_found() {
        let spec = example_arch_spec();
        let zone = spec.zone_by_id(0).unwrap();
        assert_eq!(zone.words, vec![0, 1]);
    }

    #[test]
    fn zone_by_id_not_found() {
        let spec = example_arch_spec();
        assert!(spec.zone_by_id(99).is_none());
    }

    #[test]
    fn site_bus_by_id_found() {
        let spec = example_arch_spec();
        let bus = spec.site_bus_by_id(0).unwrap();
        assert_eq!(bus.src, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn site_bus_by_id_not_found() {
        let spec = example_arch_spec();
        assert!(spec.site_bus_by_id(99).is_none());
    }

    #[test]
    fn word_bus_by_id_found() {
        let spec = example_arch_spec();
        let bus = spec.word_bus_by_id(0).unwrap();
        assert_eq!(bus.src, vec![0]);
        assert_eq!(bus.dst, vec![1]);
    }

    #[test]
    fn word_bus_by_id_not_found() {
        let spec = example_arch_spec();
        assert!(spec.word_bus_by_id(99).is_none());
    }

    #[test]
    fn site_bus_resolve_forward() {
        let spec = example_arch_spec();
        let bus = spec.site_bus_by_id(0).unwrap();
        assert_eq!(bus.resolve_forward(0), Some(5));
        assert_eq!(bus.resolve_forward(4), Some(9));
        assert_eq!(bus.resolve_forward(99), None);
    }

    #[test]
    fn site_bus_resolve_backward() {
        let spec = example_arch_spec();
        let bus = spec.site_bus_by_id(0).unwrap();
        assert_eq!(bus.resolve_backward(5), Some(0));
        assert_eq!(bus.resolve_backward(9), Some(4));
        assert_eq!(bus.resolve_backward(99), None);
    }

    #[test]
    fn word_bus_resolve_forward() {
        let spec = example_arch_spec();
        let bus = spec.word_bus_by_id(0).unwrap();
        assert_eq!(bus.resolve_forward(0), Some(1));
        assert_eq!(bus.resolve_forward(99), None);
    }

    #[test]
    fn word_bus_resolve_backward() {
        let spec = example_arch_spec();
        let bus = spec.word_bus_by_id(0).unwrap();
        assert_eq!(bus.resolve_backward(1), Some(0));
        assert_eq!(bus.resolve_backward(99), None);
    }

    #[test]
    fn site_position_valid() {
        let spec = example_arch_spec();
        let word = spec.word_by_id(0).unwrap();
        // Site 0 is [0, 0] -> x_positions[0]=1.0, y_positions[0]=2.5
        assert_eq!(word.site_position(0), Some((1.0, 2.5)));
        // Site 5 is [0, 1] -> x_positions[0]=1.0, y_positions[1]=5.0
        assert_eq!(word.site_position(5), Some((1.0, 5.0)));
        // Site 4 is [4, 0] -> x_positions[4]=9.0, y_positions[0]=2.5
        assert_eq!(word.site_position(4), Some((9.0, 2.5)));
    }

    #[test]
    fn site_position_out_of_range() {
        let spec = example_arch_spec();
        let word = spec.word_by_id(0).unwrap();
        assert!(word.site_position(99).is_none());
    }

    #[test]
    fn location_position_valid() {
        let spec = example_arch_spec();
        let loc = crate::arch::addr::LocationAddr {
            word_id: 0,
            site_id: 0,
        };
        assert_eq!(spec.location_position(&loc), Some((1.0, 2.5)));
    }

    #[test]
    fn location_position_invalid_word() {
        let spec = example_arch_spec();
        let loc = crate::arch::addr::LocationAddr {
            word_id: 99,
            site_id: 0,
        };
        assert!(spec.location_position(&loc).is_none());
    }

    #[test]
    fn location_position_invalid_site() {
        let spec = example_arch_spec();
        let loc = crate::arch::addr::LocationAddr {
            word_id: 0,
            site_id: 99,
        };
        assert!(spec.location_position(&loc).is_none());
    }

    #[test]
    fn from_json_valid() {
        let json = serde_json::to_string(&example_arch_spec()).unwrap();
        let spec = super::super::ArchSpec::from_json(&json).unwrap();
        assert_eq!(spec.version, crate::version::Version::new(1, 0));
    }

    #[test]
    fn from_json_validated_valid() {
        let json = serde_json::to_string(&example_arch_spec()).unwrap();
        let spec = super::super::ArchSpec::from_json_validated(&json).unwrap();
        assert_eq!(spec.version, crate::version::Version::new(1, 0));
    }

    #[test]
    fn check_lane_group_consistency_empty() {
        let spec = example_arch_spec();
        assert!(spec.check_lane_group_consistency(&[]).is_empty());
    }

    #[test]
    fn from_json_validated_invalid() {
        // Missing required fields
        let json = r#"{"version": "1.0"}"#;
        let result = super::super::ArchSpec::from_json_validated(json);
        assert!(result.is_err());
    }
}
