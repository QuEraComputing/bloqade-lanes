//! Arch spec queries: JSON loading, position lookup, lane resolution,
//! and group-level address validation.

use std::collections::{HashMap, HashSet};
use std::fmt;

use thiserror::Error;

use super::addr::{Direction, LaneAddr, LocationAddr, MoveType, SiteRef, WordRef, ZonedWordRef};
use super::types::{ArchSpec, Bus, Word, Zone};
use super::validate::ArchSpecError;

/// Error returned when loading an arch spec from JSON fails.
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
    DuplicateAddress { address: u64 },
    /// A location address is invalid per the arch spec.
    InvalidAddress {
        zone_id: u32,
        word_id: u32,
        site_id: u32,
    },
}

impl fmt::Display for LocationGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocationGroupError::DuplicateAddress { address } => {
                let addr = LocationAddr::decode(*address);
                write!(
                    f,
                    "duplicate location address zone_id={}, word_id={}, site_id={}",
                    addr.zone_id, addr.word_id, addr.site_id
                )
            }
            LocationGroupError::InvalidAddress {
                zone_id,
                word_id,
                site_id,
            } => {
                write!(
                    f,
                    "invalid location zone_id={}, word_id={}, site_id={}",
                    zone_id, word_id, site_id
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
    /// Lanes have inconsistent bus_id, move_type, direction, or zone_id.
    Inconsistent { message: String },
    /// Lane word_id not in zone's words_with_site_buses.
    WordNotInSiteBusList { zone_id: u32, word_id: u32 },
    /// Lane site_id not in zone's sites_with_word_buses.
    SiteNotInWordBusList { zone_id: u32, site_id: u32 },
    /// Lane group violates AOD grid constraint (e.g. not a complete grid).
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
            LaneGroupError::WordNotInSiteBusList { zone_id, word_id } => {
                write!(
                    f,
                    "zone {}: word_id {} not in words_with_site_buses",
                    zone_id, word_id
                )
            }
            LaneGroupError::SiteNotInWordBusList { zone_id, site_id } => {
                write!(
                    f,
                    "zone {}: site_id {} not in sites_with_word_buses",
                    zone_id, site_id
                )
            }
            LaneGroupError::AODConstraintViolation { message } => {
                write!(f, "AOD constraint violation: {}", message)
            }
        }
    }
}

impl std::error::Error for LaneGroupError {}

// --- Bus resolve methods ---

impl Bus<SiteRef> {
    /// Given a source site, return the destination site (forward move).
    pub fn resolve_forward(&self, src: u16) -> Option<u16> {
        self.src
            .iter()
            .position(|s| s.0 == src)
            .and_then(|i| self.dst.get(i).map(|d| d.0))
    }

    /// Given a destination site, return the source site (backward move).
    pub fn resolve_backward(&self, dst: u16) -> Option<u16> {
        self.dst
            .iter()
            .position(|d| d.0 == dst)
            .and_then(|i| self.src.get(i).map(|s| s.0))
    }
}

impl Bus<WordRef> {
    /// Given a source word, return the destination word (forward move).
    pub fn resolve_forward(&self, src: u16) -> Option<u16> {
        self.src
            .iter()
            .position(|s| s.0 == src)
            .and_then(|i| self.dst.get(i).map(|d| d.0))
    }

    /// Given a destination word, return the source word (backward move).
    pub fn resolve_backward(&self, dst: u16) -> Option<u16> {
        self.dst
            .iter()
            .position(|d| d.0 == dst)
            .and_then(|i| self.src.get(i).map(|s| s.0))
    }
}

impl Bus<ZonedWordRef> {
    /// Given a source ZonedWordRef, return the destination (forward move).
    pub fn resolve_forward(&self, src: &ZonedWordRef) -> Option<&ZonedWordRef> {
        self.src
            .iter()
            .position(|s| s == src)
            .and_then(|i| self.dst.get(i))
    }

    /// Given a destination ZonedWordRef, return the source (backward move).
    pub fn resolve_backward(&self, dst: &ZonedWordRef) -> Option<&ZonedWordRef> {
        self.dst
            .iter()
            .position(|d| d == dst)
            .and_then(|i| self.src.get(i))
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

    /// Look up a word by its index.
    pub fn word_by_id(&self, id: u32) -> Option<&Word> {
        self.words.get(id as usize)
    }

    /// Look up a zone by its index.
    pub fn zone_by_id(&self, id: u32) -> Option<&Zone> {
        self.zones.get(id as usize)
    }

    // -- Derived topology queries --

    /// Build a bidirectional word-partner map from all zones' entangling pairs.
    ///
    /// For each `[w_a, w_b]` pair in any zone, the map contains both
    /// `w_a -> w_b` and `w_b -> w_a`. Words not appearing in any pair
    /// are absent from the map.
    pub fn word_partner_map(&self) -> HashMap<u32, u32> {
        let mut map = HashMap::new();
        for zone in &self.zones {
            for &[w_a, w_b] in &zone.entangling_pairs {
                map.insert(w_a, w_b);
                map.insert(w_b, w_a);
            }
        }
        map
    }

    /// Map each word to the zone that owns it.
    ///
    /// Derived from each zone's `entangling_pairs`, `word_buses`, and
    /// `words_with_site_buses`. First match wins. Words not referenced
    /// by any zone default to zone 0.
    pub fn word_zone_map(&self) -> HashMap<u32, u32> {
        let mut map = HashMap::new();
        for (zone_id, zone) in self.zones.iter().enumerate() {
            let zid = zone_id as u32;
            for &[w_a, w_b] in &zone.entangling_pairs {
                map.entry(w_a).or_insert(zid);
                map.entry(w_b).or_insert(zid);
            }
            for bus in &zone.word_buses {
                for wref in &bus.src {
                    map.entry(wref.0 as u32).or_insert(zid);
                }
                for wref in &bus.dst {
                    map.entry(wref.0 as u32).or_insert(zid);
                }
            }
            for &wid in &zone.words_with_site_buses {
                map.entry(wid).or_insert(zid);
            }
        }
        for wid in 0..self.words.len() as u32 {
            map.entry(wid).or_insert(0);
        }
        map
    }

    /// Return the set of "home" word IDs — the lower word in each entangling
    /// pair, plus any word not appearing in any pair.
    pub fn left_cz_word_ids(&self) -> Vec<u32> {
        let partner = self.word_partner_map();
        let mut paired: HashSet<u32> = HashSet::new();
        let mut home: HashSet<u32> = HashSet::new();
        for (&w_a, &w_b) in &partner {
            paired.insert(w_a);
            paired.insert(w_b);
            home.insert(w_a.min(w_b));
        }
        for wid in 0..self.words.len() as u32 {
            if !paired.contains(&wid) {
                home.insert(wid);
            }
        }
        let mut result: Vec<u32> = home.into_iter().collect();
        result.sort();
        result
    }

    /// Reverse-lookup: given (src, dst) location pair, find the LaneAddr
    /// that connects them (if any).
    ///
    /// Searches SiteBus, WordBus, and ZoneBus lanes. The search is
    /// narrowed by exploiting the LaneAddr encoding: the
    /// `(zone_id, word_id, site_id)` in a lane address correspond to the
    /// move's source location (Forward) or destination location (Backward).
    /// So given `(src, dst)` we only iterate over `bus_id × move_type`
    /// for each direction — typically <20 candidates total — rather than
    /// enumerating every lane in the architecture.
    ///
    /// Membership lists (`words_with_site_buses`, `sites_with_word_buses`)
    /// further prune: if the candidate word/site isn't in the relevant
    /// list, that move type is skipped entirely.
    pub fn lane_for_endpoints(&self, src: &LocationAddr, dst: &LocationAddr) -> Option<LaneAddr> {
        // Try Forward: lane address fields come from src.
        if let Some(lane) = self.try_lane_from_location(src, dst, Direction::Forward) {
            return Some(lane);
        }
        // Try Backward: lane address fields come from dst.
        self.try_lane_from_location(dst, src, Direction::Backward)
    }

    /// Helper for `lane_for_endpoints`: given the location that defines
    /// the lane address fields (`origin`) and the expected other endpoint
    /// (`target`), try each bus_id × move_type combination.
    fn try_lane_from_location(
        &self,
        origin: &LocationAddr,
        target: &LocationAddr,
        direction: Direction,
    ) -> Option<LaneAddr> {
        let zone = self.zones.get(origin.zone_id as usize)?;

        // SiteBus: only if origin's word is in words_with_site_buses.
        if zone.words_with_site_buses.contains(&origin.word_id) {
            for bus_id in 0..zone.site_buses.len() {
                if let Some(lane) = self.check_lane_candidate(
                    MoveType::SiteBus,
                    origin,
                    target,
                    bus_id as u32,
                    direction,
                ) {
                    return Some(lane);
                }
            }
        }

        // WordBus: only if origin's site is in sites_with_word_buses.
        if zone.sites_with_word_buses.contains(&origin.site_id) {
            for bus_id in 0..zone.word_buses.len() {
                if let Some(lane) = self.check_lane_candidate(
                    MoveType::WordBus,
                    origin,
                    target,
                    bus_id as u32,
                    direction,
                ) {
                    return Some(lane);
                }
            }
        }

        // ZoneBus: buses live on self (not per-zone).
        for bus_id in 0..self.zone_buses.len() {
            if let Some(lane) = self.check_lane_candidate(
                MoveType::ZoneBus,
                origin,
                target,
                bus_id as u32,
                direction,
            ) {
                return Some(lane);
            }
        }

        None
    }

    /// Construct a candidate `LaneAddr` from the origin location and
    /// check whether its resolved endpoints match `(origin, target)`.
    fn check_lane_candidate(
        &self,
        move_type: MoveType,
        origin: &LocationAddr,
        target: &LocationAddr,
        bus_id: u32,
        direction: Direction,
    ) -> Option<LaneAddr> {
        let lane = LaneAddr {
            move_type,
            zone_id: origin.zone_id,
            word_id: origin.word_id,
            site_id: origin.site_id,
            bus_id,
            direction,
        };
        let (s, d) = self.lane_endpoints(&lane)?;
        let (expected_src, expected_dst) = match direction {
            Direction::Forward => (s, d),
            Direction::Backward => (d, s),
        };
        if expected_src == *origin && expected_dst == *target {
            Some(lane)
        } else {
            None
        }
    }

    // -- Position resolution --

    /// Resolve a LocationAddr to physical (x, y) coordinates.
    ///
    /// Uses the zone's grid and the word's site index pair to compute
    /// the physical position.
    pub fn location_position(&self, loc: &LocationAddr) -> Option<(f64, f64)> {
        let zone = self.zones.get(loc.zone_id as usize)?;
        let word = self.words.get(loc.word_id as usize)?;
        let site = word.sites.get(loc.site_id as usize)?;
        let x = zone.grid.x_position(site[0] as usize)?;
        let y = zone.grid.y_position(site[1] as usize)?;
        Some((x, y))
    }

    /// Resolve a `LaneAddr` to its source and destination `LocationAddr` pair.
    ///
    /// Returns `Some((src, dst))` if the lane can be resolved through the bus,
    /// or `None` if the lane references invalid zones, words, sites, or buses.
    pub fn lane_endpoints(&self, lane: &LaneAddr) -> Option<(LocationAddr, LocationAddr)> {
        // Validate the lane address up front so callers always get None
        // for invalid lanes (e.g. out-of-range zone_id, word_id, or site_id).
        if !self.check_lane(lane).is_empty() {
            return None;
        }

        let zone = self.zone_by_id(lane.zone_id)?;

        // In the lane address convention, site_id and word_id always encode
        // the forward-direction source. The direction field only controls
        // which endpoint is returned as src vs dst.
        let fwd_src = LocationAddr {
            zone_id: lane.zone_id,
            word_id: lane.word_id,
            site_id: lane.site_id,
        };

        let fwd_dst = match lane.move_type {
            MoveType::SiteBus => {
                let bus = zone.site_buses.get(lane.bus_id as usize)?;
                let dst_site = bus.resolve_forward(lane.site_id as u16)?;
                LocationAddr {
                    zone_id: lane.zone_id,
                    word_id: lane.word_id,
                    site_id: dst_site as u32,
                }
            }
            MoveType::WordBus => {
                let bus = zone.word_buses.get(lane.bus_id as usize)?;
                let dst_word = bus.resolve_forward(lane.word_id as u16)?;
                LocationAddr {
                    zone_id: lane.zone_id,
                    word_id: dst_word as u32,
                    site_id: lane.site_id,
                }
            }
            MoveType::ZoneBus => {
                let bus = self.zone_buses.get(lane.bus_id as usize)?;
                let src_ref = ZonedWordRef {
                    zone_id: lane.zone_id as u8,
                    word_id: lane.word_id as u16,
                };
                let dst_ref = bus.resolve_forward(&src_ref)?;
                LocationAddr {
                    zone_id: dst_ref.zone_id as u32,
                    word_id: dst_ref.word_id as u32,
                    site_id: lane.site_id,
                }
            }
        };

        match lane.direction {
            Direction::Forward => Some((fwd_src, fwd_dst)),
            Direction::Backward => Some((fwd_dst, fwd_src)),
        }
    }

    /// Get the CZ partner for a given location.
    ///
    /// Searches `zones[loc.zone_id].entangling_pairs` for a pair containing
    /// `loc.word_id`. Returns the partner in the **same zone** with the paired
    /// word_id and same site_id. Returns `None` if the word is not in any
    /// entangling pair within its zone.
    pub fn get_cz_partner(&self, loc: &LocationAddr) -> Option<LocationAddr> {
        let zone = self.zones.get(loc.zone_id as usize)?;
        let partner_word = zone.entangling_pairs.iter().find_map(|pair| {
            if pair[0] == loc.word_id {
                Some(pair[1])
            } else if pair[1] == loc.word_id {
                Some(pair[0])
            } else {
                None
            }
        })?;
        Some(LocationAddr {
            zone_id: loc.zone_id,
            word_id: partner_word,
            site_id: loc.site_id,
        })
    }

    // -- Address validation --

    /// Check whether a location address (zone_id, word_id, site_id) is valid.
    pub fn check_location(&self, loc: &LocationAddr) -> Option<String> {
        let num_zones = self.zones.len() as u32;
        let num_words = self.words.len() as u32;
        let sites_per_word = self.sites_per_word() as u32;

        if loc.zone_id >= num_zones {
            return Some(format!(
                "invalid location zone_id={} (num_zones={})",
                loc.zone_id, num_zones
            ));
        }
        if loc.word_id >= num_words {
            return Some(format!(
                "invalid location word_id={} (num_words={})",
                loc.word_id, num_words
            ));
        }
        if loc.site_id >= sites_per_word {
            return Some(format!(
                "invalid location site_id={} (sites_per_word={})",
                loc.site_id, sites_per_word
            ));
        }
        None
    }

    /// Check whether a lane address is valid.
    ///
    /// Validates that the zone and bus exist, word/site are in range, and the
    /// site/word is a valid forward source for the bus. For SiteBus/WordBus,
    /// buses are looked up from the zone. For ZoneBus, buses are looked up
    /// from `self.zone_buses`.
    pub fn check_lane(&self, addr: &LaneAddr) -> Vec<String> {
        let num_zones = self.zones.len() as u32;
        let num_words = self.words.len() as u32;
        let sites_per_word = self.sites_per_word() as u32;
        let mut errors = Vec::new();

        // Validate zone_id first since other checks depend on it
        if addr.zone_id >= num_zones {
            errors.push(format!(
                "zone_id {} out of range (num_zones={})",
                addr.zone_id, num_zones
            ));
            return errors;
        }

        let zone = &self.zones[addr.zone_id as usize];

        match addr.move_type {
            MoveType::SiteBus => {
                if addr.word_id >= num_words {
                    errors.push(format!("word_id {} out of range", addr.word_id));
                }
                if addr.site_id >= sites_per_word {
                    errors.push(format!("site_id {} out of range", addr.site_id));
                }
                if let Some(bus) = zone.site_buses.get(addr.bus_id as usize) {
                    if addr.word_id < num_words
                        && !zone.words_with_site_buses.contains(&addr.word_id)
                    {
                        errors.push(format!(
                            "word_id {} not in zone {} words_with_site_buses",
                            addr.word_id, addr.zone_id
                        ));
                    }
                    if errors.is_empty() && bus.resolve_forward(addr.site_id as u16).is_none() {
                        errors.push(format!(
                            "site_id {} is not a valid source for zone {} site_bus {}",
                            addr.site_id, addr.zone_id, addr.bus_id
                        ));
                    }
                } else {
                    errors.push(format!(
                        "unknown site_bus id {} in zone {}",
                        addr.bus_id, addr.zone_id
                    ));
                }
            }
            MoveType::WordBus => {
                if addr.word_id >= num_words {
                    errors.push(format!("word_id {} out of range", addr.word_id));
                }
                if addr.site_id >= sites_per_word {
                    errors.push(format!("site_id {} out of range", addr.site_id));
                } else if !zone.sites_with_word_buses.contains(&addr.site_id) {
                    errors.push(format!(
                        "site_id {} not in zone {} sites_with_word_buses",
                        addr.site_id, addr.zone_id
                    ));
                }
                if let Some(bus) = zone.word_buses.get(addr.bus_id as usize) {
                    if errors.is_empty() && bus.resolve_forward(addr.word_id as u16).is_none() {
                        errors.push(format!(
                            "word_id {} is not a valid source for zone {} word_bus {}",
                            addr.word_id, addr.zone_id, addr.bus_id
                        ));
                    }
                } else {
                    errors.push(format!(
                        "unknown word_bus id {} in zone {}",
                        addr.bus_id, addr.zone_id
                    ));
                }
            }
            MoveType::ZoneBus => {
                if addr.word_id >= num_words {
                    errors.push(format!("word_id {} out of range", addr.word_id));
                }
                if addr.site_id >= sites_per_word {
                    errors.push(format!("site_id {} out of range", addr.site_id));
                }
                if let Some(bus) = self.zone_buses.get(addr.bus_id as usize) {
                    let src_ref = ZonedWordRef {
                        zone_id: addr.zone_id as u8,
                        word_id: addr.word_id as u16,
                    };
                    if errors.is_empty() && bus.resolve_forward(&src_ref).is_none() {
                        errors.push(format!(
                            "zone_id={}, word_id={} is not a valid source for zone_bus {}",
                            addr.zone_id, addr.word_id, addr.bus_id
                        ));
                    }
                } else {
                    errors.push(format!("unknown zone_bus id {}", addr.bus_id));
                }
            }
        }
        errors
    }

    /// Check whether a zone address is valid.
    pub fn check_zone(&self, zone: &super::addr::ZoneAddr) -> Option<String> {
        if self.zone_by_id(zone.zone_id).is_none() {
            Some(format!("invalid zone_id={}", zone.zone_id))
        } else {
            None
        }
    }

    // -- Group validation --

    /// Check that a group of lanes share consistent bus_id, move_type, direction, and zone_id.
    pub fn check_lane_group_consistency(&self, lanes: &[LaneAddr]) -> Vec<String> {
        if lanes.is_empty() {
            return vec![];
        }
        let first = &lanes[0];
        let mut errors = Vec::new();

        for lane in &lanes[1..] {
            if lane.zone_id != first.zone_id {
                errors.push(format!(
                    "zone_id mismatch: expected {}, got {}",
                    first.zone_id, lane.zone_id
                ));
            }
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

    /// Check that each lane's word/site belongs to the correct zone's bus membership list.
    ///
    /// For SiteBus, checks zone's `words_with_site_buses`.
    /// For WordBus, checks zone's `sites_with_word_buses`.
    /// ZoneBus has no membership list (zone buses are global).
    ///
    /// Returns unique `(word_ids_not_in_site_bus_list, site_ids_not_in_word_bus_list)`.
    pub fn check_lane_group_membership(&self, lanes: &[LaneAddr]) -> (Vec<u32>, Vec<u32>) {
        use std::collections::BTreeSet;

        let mut bad_words = BTreeSet::new();
        let mut bad_sites = BTreeSet::new();

        for lane in lanes {
            let zone = match self.zones.get(lane.zone_id as usize) {
                Some(z) => z,
                None => continue, // zone validation handled elsewhere
            };

            match lane.move_type {
                MoveType::SiteBus => {
                    if !zone.words_with_site_buses.contains(&lane.word_id) {
                        bad_words.insert(lane.word_id);
                    }
                }
                MoveType::WordBus => {
                    if !zone.sites_with_word_buses.contains(&lane.site_id) {
                        bad_sites.insert(lane.site_id);
                    }
                }
                MoveType::ZoneBus => {
                    // Zone buses are global; no per-zone membership list.
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
                    zone_id: loc.zone_id,
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
            // Use the first lane's zone_id for error context (consistency already checked)
            let zone_id = lanes[0].zone_id;
            for word_id in bad_words {
                errors.push(LaneGroupError::WordNotInSiteBusList { zone_id, word_id });
            }
            for site_id in bad_sites {
                errors.push(LaneGroupError::SiteNotInWordBusList { zone_id, site_id });
            }
            for msg in self.check_lane_group_geometry(lanes) {
                errors.push(LaneGroupError::AODConstraintViolation { message: msg });
            }
        }

        errors
    }

    /// Check AOD grid constraint: lane positions must form a complete grid
    /// (Cartesian product of unique X and Y values).
    pub fn check_lane_group_geometry(&self, lanes: &[LaneAddr]) -> Vec<String> {
        use std::collections::BTreeSet;

        let positions: Vec<(f64, f64)> = lanes
            .iter()
            .filter_map(|lane| {
                let loc = LocationAddr {
                    zone_id: lane.zone_id,
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
                "lanes do not form a complete grid: expected {} positions ({}x * {}y), got {} unique positions",
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
    use super::*;
    use crate::arch::addr::{
        Direction, LaneAddr, LocationAddr, MoveType, SiteRef, WordRef, ZoneAddr, ZonedWordRef,
    };
    use crate::arch::types::{Grid, Mode};
    use crate::version::Version;

    /// Create a valid two-zone arch spec for testing.
    /// Mirrors the helper in validate.rs tests.
    fn make_valid_two_zone_spec() -> ArchSpec {
        let grid0 = Grid::from_positions(&[0.0, 5.0, 10.0], &[0.0, 3.0]);
        // Zone 1 grid must not overlap zone 0 (x=[0,10], y=[0,3]).
        let grid1 = Grid::from_positions(&[20.0, 27.5, 35.0], &[0.0, 4.0]);

        ArchSpec {
            version: Version::new(2, 0),
            words: vec![
                Word {
                    sites: vec![[0, 0], [0, 1]],
                },
                Word {
                    sites: vec![[1, 0], [1, 1]],
                },
            ],
            zones: vec![
                Zone {
                    name: String::new(),
                    grid: grid0,
                    site_buses: vec![Bus {
                        src: vec![SiteRef(0)],
                        dst: vec![SiteRef(1)],
                    }],
                    word_buses: vec![Bus {
                        src: vec![WordRef(0)],
                        dst: vec![WordRef(1)],
                    }],
                    words_with_site_buses: vec![0, 1],
                    sites_with_word_buses: vec![0],
                    entangling_pairs: vec![[0, 1]],
                },
                Zone {
                    name: String::new(),
                    grid: grid1,
                    site_buses: vec![],
                    word_buses: vec![],
                    words_with_site_buses: vec![],
                    sites_with_word_buses: vec![],
                    entangling_pairs: vec![],
                },
            ],
            zone_buses: vec![Bus {
                src: vec![ZonedWordRef {
                    zone_id: 0,
                    word_id: 0,
                }],
                dst: vec![ZonedWordRef {
                    zone_id: 1,
                    word_id: 0,
                }],
            }],
            modes: vec![Mode {
                name: "full".to_string(),
                zones: vec![0, 1],
                bitstring_order: vec![],
            }],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
        }
    }

    // ── location_position tests ──

    #[test]
    fn test_location_position_zone0() {
        let spec = make_valid_two_zone_spec();
        // Zone 0 grid: x=[0.0, 5.0, 10.0] y=[0.0, 3.0]
        // Word 0: sites=[(0,0), (0,1)] -> site 0 at grid[0][0] = (0.0, 0.0)
        let pos = spec.location_position(&LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        });
        assert_eq!(pos, Some((0.0, 0.0)));
    }

    #[test]
    fn test_location_position_zone0_site1() {
        let spec = make_valid_two_zone_spec();
        // Word 0: sites=[(0,0), (0,1)] -> site 1 at grid x[0]=0.0, y[1]=3.0
        let pos = spec.location_position(&LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 1,
        });
        assert_eq!(pos, Some((0.0, 3.0)));
    }

    #[test]
    fn test_location_position_zone1() {
        let spec = make_valid_two_zone_spec();
        // Zone 1 grid: x=[20.0, 27.5, 35.0] y=[0.0, 4.0]
        // Word 1: sites=[(1,0), (1,1)] -> site 0 at grid x[1]=27.5, y[0]=0.0
        let pos = spec.location_position(&LocationAddr {
            zone_id: 1,
            word_id: 1,
            site_id: 0,
        });
        assert_eq!(pos, Some((27.5, 0.0)));
    }

    #[test]
    fn test_location_position_invalid_zone() {
        let spec = make_valid_two_zone_spec();
        let pos = spec.location_position(&LocationAddr {
            zone_id: 99,
            word_id: 0,
            site_id: 0,
        });
        assert!(pos.is_none());
    }

    #[test]
    fn test_location_position_invalid_word() {
        let spec = make_valid_two_zone_spec();
        let pos = spec.location_position(&LocationAddr {
            zone_id: 0,
            word_id: 99,
            site_id: 0,
        });
        assert!(pos.is_none());
    }

    #[test]
    fn test_location_position_invalid_site() {
        let spec = make_valid_two_zone_spec();
        let pos = spec.location_position(&LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 99,
        });
        assert!(pos.is_none());
    }

    // ── get_cz_partner tests ──

    #[test]
    fn test_get_cz_partner() {
        let spec = make_valid_two_zone_spec();
        // Zone 0 has entangling_pairs: [[0, 1]] — word 0 paired with word 1
        let partner = spec.get_cz_partner(&LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        });
        assert_eq!(
            partner,
            Some(LocationAddr {
                zone_id: 0, // same zone
                word_id: 1, // partner word
                site_id: 0,
            })
        );
    }

    #[test]
    fn test_get_cz_partner_reverse() {
        let spec = make_valid_two_zone_spec();
        // word 1 → word 0 (reverse direction within same zone)
        let partner = spec.get_cz_partner(&LocationAddr {
            zone_id: 0,
            word_id: 1,
            site_id: 1,
        });
        assert_eq!(
            partner,
            Some(LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1,
            })
        );
    }

    #[test]
    fn test_get_cz_partner_no_pair() {
        let spec = make_valid_two_zone_spec();
        // Zone 1 has no entangling pairs
        let partner = spec.get_cz_partner(&LocationAddr {
            zone_id: 1,
            word_id: 0,
            site_id: 0,
        });
        assert!(partner.is_none());
    }

    // ── lane_endpoints tests ──

    #[test]
    fn test_lane_endpoints_site_bus() {
        let spec = make_valid_two_zone_spec();
        // Zone 0 has site_bus: src=[SiteRef(0)] dst=[SiteRef(1)]
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let (src, dst) = spec.lane_endpoints(&lane).unwrap();
        assert_eq!(
            src,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            }
        );
        assert_eq!(
            dst,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1,
            }
        );
    }

    #[test]
    fn test_lane_endpoints_site_bus_backward() {
        let spec = make_valid_two_zone_spec();
        let lane = LaneAddr {
            direction: Direction::Backward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let (src, dst) = spec.lane_endpoints(&lane).unwrap();
        // Backward swaps: src is forward dst, dst is forward src
        assert_eq!(
            src,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1,
            }
        );
        assert_eq!(
            dst,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            }
        );
    }

    #[test]
    fn test_lane_endpoints_word_bus() {
        let spec = make_valid_two_zone_spec();
        // Zone 0 has word_bus: src=[WordRef(0)] dst=[WordRef(1)]
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::WordBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let (src, dst) = spec.lane_endpoints(&lane).unwrap();
        assert_eq!(
            src,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            }
        );
        assert_eq!(
            dst,
            LocationAddr {
                zone_id: 0,
                word_id: 1,
                site_id: 0,
            }
        );
    }

    #[test]
    fn test_lane_endpoints_zone_bus() {
        let spec = make_valid_two_zone_spec();
        // zone_bus: src=[ZWR(0,0)] dst=[ZWR(1,0)]
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::ZoneBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let (src, dst) = spec.lane_endpoints(&lane).unwrap();
        assert_eq!(
            src,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            }
        );
        assert_eq!(
            dst,
            LocationAddr {
                zone_id: 1,
                word_id: 0,
                site_id: 0,
            }
        );
    }

    #[test]
    fn test_lane_endpoints_invalid_bus_returns_none() {
        let spec = make_valid_two_zone_spec();
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 99,
        };
        assert!(spec.lane_endpoints(&lane).is_none());
    }

    // ── JSON round-trip tests ──

    #[test]
    fn test_json_round_trip() {
        let spec = make_valid_two_zone_spec();
        let json = serde_json::to_string_pretty(&spec).unwrap();
        let deserialized = ArchSpec::from_json(&json).unwrap();
        assert_eq!(spec, deserialized);
    }

    #[test]
    fn test_from_json_validated() {
        let spec = make_valid_two_zone_spec();
        let json = serde_json::to_string(&spec).unwrap();
        let validated = ArchSpec::from_json_validated(&json).unwrap();
        assert_eq!(spec, validated);
    }

    #[test]
    fn test_from_json_validated_invalid() {
        let json = r#"{"version": "1.0"}"#;
        let result = ArchSpec::from_json_validated(json);
        assert!(result.is_err());
    }

    // ── word/zone lookup tests ──

    #[test]
    fn test_word_by_id_found() {
        let spec = make_valid_two_zone_spec();
        let word = spec.word_by_id(0).unwrap();
        assert_eq!(word.sites.len(), 2);
    }

    #[test]
    fn test_word_by_id_not_found() {
        let spec = make_valid_two_zone_spec();
        assert!(spec.word_by_id(99).is_none());
    }

    #[test]
    fn test_zone_by_id_found() {
        let spec = make_valid_two_zone_spec();
        let zone = spec.zone_by_id(0).unwrap();
        assert_eq!(zone.site_buses.len(), 1);
    }

    #[test]
    fn test_zone_by_id_not_found() {
        let spec = make_valid_two_zone_spec();
        assert!(spec.zone_by_id(99).is_none());
    }

    // ── Bus resolve tests ──

    #[test]
    fn test_site_bus_resolve_forward() {
        let spec = make_valid_two_zone_spec();
        let bus = &spec.zones[0].site_buses[0];
        assert_eq!(bus.resolve_forward(0), Some(1));
        assert_eq!(bus.resolve_forward(99), None);
    }

    #[test]
    fn test_site_bus_resolve_backward() {
        let spec = make_valid_two_zone_spec();
        let bus = &spec.zones[0].site_buses[0];
        assert_eq!(bus.resolve_backward(1), Some(0));
        assert_eq!(bus.resolve_backward(99), None);
    }

    #[test]
    fn test_word_bus_resolve_forward() {
        let spec = make_valid_two_zone_spec();
        let bus = &spec.zones[0].word_buses[0];
        assert_eq!(bus.resolve_forward(0), Some(1));
        assert_eq!(bus.resolve_forward(99), None);
    }

    #[test]
    fn test_word_bus_resolve_backward() {
        let spec = make_valid_two_zone_spec();
        let bus = &spec.zones[0].word_buses[0];
        assert_eq!(bus.resolve_backward(1), Some(0));
        assert_eq!(bus.resolve_backward(99), None);
    }

    #[test]
    fn test_zone_bus_resolve_forward() {
        let spec = make_valid_two_zone_spec();
        let bus = &spec.zone_buses[0];
        let src = ZonedWordRef {
            zone_id: 0,
            word_id: 0,
        };
        let dst = bus.resolve_forward(&src).unwrap();
        assert_eq!(dst.zone_id, 1);
        assert_eq!(dst.word_id, 0);
    }

    #[test]
    fn test_zone_bus_resolve_backward() {
        let spec = make_valid_two_zone_spec();
        let bus = &spec.zone_buses[0];
        let dst = ZonedWordRef {
            zone_id: 1,
            word_id: 0,
        };
        let src = bus.resolve_backward(&dst).unwrap();
        assert_eq!(src.zone_id, 0);
        assert_eq!(src.word_id, 0);
    }

    // ── check_location tests ──

    #[test]
    fn test_check_location_valid() {
        let spec = make_valid_two_zone_spec();
        assert!(
            spec.check_location(&LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            })
            .is_none()
        );
    }

    #[test]
    fn test_check_location_invalid_zone() {
        let spec = make_valid_two_zone_spec();
        let err = spec
            .check_location(&LocationAddr {
                zone_id: 99,
                word_id: 0,
                site_id: 0,
            })
            .unwrap();
        assert!(err.contains("zone_id"));
    }

    // ── check_lane tests ──

    #[test]
    fn test_check_lane_valid_site_bus() {
        let spec = make_valid_two_zone_spec();
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        assert!(spec.check_lane(&lane).is_empty());
    }

    #[test]
    fn test_check_lane_invalid_zone() {
        let spec = make_valid_two_zone_spec();
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 99,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let errors = spec.check_lane(&lane);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("zone_id"));
    }

    #[test]
    fn test_check_lane_invalid_bus() {
        let spec = make_valid_two_zone_spec();
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 99,
        };
        let errors = spec.check_lane(&lane);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_check_lane_zone_bus_valid() {
        let spec = make_valid_two_zone_spec();
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::ZoneBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        assert!(spec.check_lane(&lane).is_empty());
    }

    #[test]
    fn test_check_lane_zone_bus_invalid_bus() {
        let spec = make_valid_two_zone_spec();
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::ZoneBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 99,
        };
        let errors = spec.check_lane(&lane);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("zone_bus"));
    }

    // ── check_zone tests ──

    #[test]
    fn test_check_zone_valid() {
        let spec = make_valid_two_zone_spec();
        assert!(spec.check_zone(&ZoneAddr { zone_id: 0 }).is_none());
    }

    #[test]
    fn test_check_zone_invalid() {
        let spec = make_valid_two_zone_spec();
        assert!(spec.check_zone(&ZoneAddr { zone_id: 99 }).is_some());
    }

    // ── check_lane_group_consistency tests ──

    #[test]
    fn test_check_lane_group_consistency_empty() {
        let spec = make_valid_two_zone_spec();
        assert!(spec.check_lane_group_consistency(&[]).is_empty());
    }

    #[test]
    fn test_check_lane_group_consistency_zone_mismatch() {
        let spec = make_valid_two_zone_spec();
        let lanes = vec![
            LaneAddr {
                direction: Direction::Forward,
                move_type: MoveType::SiteBus,
                zone_id: 0,
                word_id: 0,
                site_id: 0,
                bus_id: 0,
            },
            LaneAddr {
                direction: Direction::Forward,
                move_type: MoveType::SiteBus,
                zone_id: 1,
                word_id: 0,
                site_id: 0,
                bus_id: 0,
            },
        ];
        let errors = spec.check_lane_group_consistency(&lanes);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("zone_id mismatch"));
    }

    // ── check_locations tests ──

    #[test]
    fn test_check_locations_valid() {
        let spec = make_valid_two_zone_spec();
        let locs = vec![
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1,
            },
        ];
        assert!(spec.check_locations(&locs).is_empty());
    }

    #[test]
    fn test_check_locations_duplicate() {
        let spec = make_valid_two_zone_spec();
        let locs = vec![
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        ];
        let errors = spec.check_locations(&locs);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, LocationGroupError::DuplicateAddress { .. }))
        );
    }

    #[test]
    fn test_check_locations_invalid() {
        let spec = make_valid_two_zone_spec();
        let locs = vec![LocationAddr {
            zone_id: 99,
            word_id: 0,
            site_id: 0,
        }];
        let errors = spec.check_locations(&locs);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, LocationGroupError::InvalidAddress { .. }))
        );
    }

    // ── Derived topology query tests (#464 phase 2) ──

    #[test]
    fn test_word_partner_map() {
        let spec = make_valid_two_zone_spec();
        let map = spec.word_partner_map();
        // Zone 0 has entangling_pairs=[[0, 1]], zone 1 has none.
        assert_eq!(map.get(&0), Some(&1));
        assert_eq!(map.get(&1), Some(&0));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_word_zone_map() {
        let spec = make_valid_two_zone_spec();
        let map = spec.word_zone_map();
        // Words 0 and 1 are referenced by zone 0 (entangling_pairs + buses).
        assert_eq!(map.get(&0), Some(&0));
        assert_eq!(map.get(&1), Some(&0));
        assert_eq!(map.len(), 2); // exactly 2 words
    }

    #[test]
    fn test_left_cz_word_ids() {
        let spec = make_valid_two_zone_spec();
        let home = spec.left_cz_word_ids();
        // Pair [0, 1] -> home word is 0. Word 1 is the staging word.
        // But there are only 2 words and they're all paired, so home = [0].
        assert_eq!(home, vec![0]);
    }

    #[test]
    fn test_lane_for_endpoints_site_bus() {
        let spec = make_valid_two_zone_spec();
        // Zone 0 has site_bus: src=[SiteRef(0)] dst=[SiteRef(1)], words_with_site_buses=[0,1].
        // For word 0, site bus maps site 0 -> site 1.
        let src = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let dst = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 1,
        };
        let lane = spec.lane_for_endpoints(&src, &dst);
        assert!(lane.is_some(), "should find a lane for (src, dst)");
        let l = lane.unwrap();
        assert_eq!(l.move_type, MoveType::SiteBus);
        assert_eq!(l.direction, Direction::Forward);
    }

    #[test]
    fn test_lane_for_endpoints_word_bus() {
        let spec = make_valid_two_zone_spec();
        // Zone 0 word_bus: src=[WordRef(0)] dst=[WordRef(1)], sites_with_word_buses=[0].
        let src = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let dst = LocationAddr {
            zone_id: 0,
            word_id: 1,
            site_id: 0,
        };
        let lane = spec.lane_for_endpoints(&src, &dst);
        assert!(lane.is_some(), "should find a word-bus lane");
        let l = lane.unwrap();
        assert_eq!(l.move_type, MoveType::WordBus);
        assert_eq!(l.direction, Direction::Forward);
    }

    #[test]
    fn test_lane_for_endpoints_not_found() {
        let spec = make_valid_two_zone_spec();
        // No lane connects word 0 site 0 to word 1 site 1 (different site ids
        // across a word bus move).
        let src = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        let dst = LocationAddr {
            zone_id: 0,
            word_id: 1,
            site_id: 1,
        };
        assert!(spec.lane_for_endpoints(&src, &dst).is_none());
    }
}
