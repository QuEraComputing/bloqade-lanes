//! Bit-packed address types for bytecode instructions.
//!
//! These types encode device-level addresses into compact integer
//! representations used in the 16-byte instruction format.

use serde::{Deserialize, Serialize};

/// Site index within a word. Matches the 16-bit site_id field in LocationAddr.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SiteRef(pub u16);

/// Word index within a zone. Matches the 16-bit word_id field in LocationAddr.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WordRef(pub u16);

/// Zone-qualified word reference for inter-zone bus entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZonedWordRef {
    pub zone_id: u8,
    pub word_id: u16,
}

/// Atom movement direction along a transport bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction {
    /// Movement from source to destination (value 0).
    Forward = 0,
    /// Movement from destination to source (value 1).
    Backward = 1,
}

/// Type of transport bus used for an atom move operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MoveType {
    /// Moves atoms between sites within a word (value 0).
    SiteBus = 0,
    /// Moves atoms between words (value 1).
    WordBus = 1,
}

/// Bit-packed atom location address (word + site).
///
/// Encodes `word_id` (16 bits) and `site_id` (16 bits) into a 32-bit word.
///
/// Layout: `[word_id:16][site_id:16]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocationAddr {
    pub word_id: u32,
    pub site_id: u32,
}

impl LocationAddr {
    /// Encode to a 32-bit packed integer.
    ///
    /// Layout: `[word_id:16][site_id:16]`
    pub fn encode(&self) -> u32 {
        ((self.word_id as u16 as u32) << 16) | (self.site_id as u16 as u32)
    }

    /// Decode a 32-bit packed integer into a `LocationAddr`.
    pub fn decode(bits: u32) -> Self {
        Self {
            word_id: (bits >> 16) & 0xFFFF,
            site_id: bits & 0xFFFF,
        }
    }
}

/// Bit-packed lane address for atom move operations.
///
/// Encodes direction (1 bit), move type (1 bit), word_id (16 bits),
/// site_id (16 bits), and bus_id (16 bits) across two 32-bit data words.
///
/// Layout:
/// - data0: `[word_id:16][site_id:16]`
/// - data1: `[dir:1][mt:1][pad:14][bus_id:16]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneAddr {
    pub direction: Direction,
    pub move_type: MoveType,
    pub word_id: u32,
    pub site_id: u32,
    pub bus_id: u32,
}

impl LaneAddr {
    /// Encode to two 32-bit data words `(data0, data1)`.
    pub fn encode(&self) -> (u32, u32) {
        let data0 = ((self.word_id as u16 as u32) << 16) | (self.site_id as u16 as u32);
        let data1 = ((self.direction as u32) << 31)
            | ((self.move_type as u32) << 30)
            | (self.bus_id as u16 as u32);
        (data0, data1)
    }

    /// Encode to a single 64-bit packed integer (`data0 | (data1 << 32)`).
    pub fn encode_u64(&self) -> u64 {
        let (d0, d1) = self.encode();
        (d0 as u64) | ((d1 as u64) << 32)
    }

    /// Decode two 32-bit data words into a `LaneAddr`.
    pub fn decode(data0: u32, data1: u32) -> Self {
        let direction = if (data1 >> 31) & 1 == 0 {
            Direction::Forward
        } else {
            Direction::Backward
        };
        let move_type = if (data1 >> 30) & 1 == 0 {
            MoveType::SiteBus
        } else {
            MoveType::WordBus
        };
        Self {
            direction,
            move_type,
            word_id: (data0 >> 16) & 0xFFFF,
            site_id: data0 & 0xFFFF,
            bus_id: data1 & 0xFFFF,
        }
    }

    /// Decode a 64-bit packed integer into a `LaneAddr`.
    pub fn decode_u64(bits: u64) -> Self {
        Self::decode(bits as u32, (bits >> 32) as u32)
    }
}

/// Bit-packed zone address.
///
/// Encodes a zone identifier (16 bits) into a 32-bit value.
///
/// Layout: `[pad:16][zone_id:16]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZoneAddr {
    pub zone_id: u32,
}

impl ZoneAddr {
    /// Encode to a 32-bit packed integer.
    pub fn encode(&self) -> u32 {
        self.zone_id as u16 as u32
    }

    /// Decode a 32-bit packed integer into a `ZoneAddr`.
    pub fn decode(bits: u32) -> Self {
        Self {
            zone_id: bits & 0xFFFF,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_addr_round_trip() {
        let addr = LocationAddr {
            word_id: 0xABCD,
            site_id: 0x1234,
        };
        let bits = addr.encode();
        assert_eq!(bits, 0xABCD_1234);
        assert_eq!(LocationAddr::decode(bits), addr);
    }

    #[test]
    fn test_location_addr_zero() {
        let addr = LocationAddr {
            word_id: 0,
            site_id: 0,
        };
        assert_eq!(addr.encode(), 0);
        assert_eq!(LocationAddr::decode(0), addr);
    }

    #[test]
    fn test_lane_addr_round_trip() {
        let addr = LaneAddr {
            direction: Direction::Backward,
            move_type: MoveType::WordBus,
            word_id: 0x1234,
            site_id: 0x5678,
            bus_id: 0x9ABC,
        };
        let (data0, data1) = addr.encode();
        assert_eq!(LaneAddr::decode(data0, data1), addr);

        // Check bit positions in data0
        assert_eq!((data0 >> 16) & 0xFFFF, 0x1234); // word_id
        assert_eq!(data0 & 0xFFFF, 0x5678); // site_id

        // Check bit positions in data1
        assert_eq!((data1 >> 31) & 1, 1); // direction = Backward
        assert_eq!((data1 >> 30) & 1, 1); // move_type = WordBus
        assert_eq!(data1 & 0xFFFF, 0x9ABC); // bus_id
    }

    #[test]
    fn test_lane_addr_forward_sitebus() {
        let addr = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            word_id: 0,
            site_id: 0,
            bus_id: 1,
        };
        let (data0, data1) = addr.encode();
        assert_eq!(data0, 0);
        assert_eq!(data1, 1);
        assert_eq!(LaneAddr::decode(data0, data1), addr);
    }

    #[test]
    fn test_lane_addr_u64_round_trip() {
        let addr = LaneAddr {
            direction: Direction::Backward,
            move_type: MoveType::WordBus,
            word_id: 1,
            site_id: 0,
            bus_id: 0,
        };
        let packed = addr.encode_u64();
        assert_eq!(LaneAddr::decode_u64(packed), addr);
    }

    #[test]
    fn test_zone_addr_round_trip() {
        let addr = ZoneAddr { zone_id: 42 };
        let bits = addr.encode();
        assert_eq!(bits, 42);
        assert_eq!(ZoneAddr::decode(bits), addr);
    }

    #[test]
    fn test_zone_addr_max() {
        let addr = ZoneAddr { zone_id: 0xFFFF };
        let bits = addr.encode();
        assert_eq!(bits, 0xFFFF);
        assert_eq!(ZoneAddr::decode(bits), addr);
    }

    #[test]
    fn test_site_ref_newtype() {
        let s = SiteRef(42);
        assert_eq!(s.0, 42);
        let json = serde_json::to_string(&s).unwrap();
        let deserialized: SiteRef = serde_json::from_str(&json).unwrap();
        assert_eq!(s, deserialized);
    }

    #[test]
    fn test_word_ref_newtype() {
        let w = WordRef(100);
        assert_eq!(w.0, 100);
        let json = serde_json::to_string(&w).unwrap();
        let deserialized: WordRef = serde_json::from_str(&json).unwrap();
        assert_eq!(w, deserialized);
    }

    #[test]
    fn test_zoned_word_ref() {
        let zwr = ZonedWordRef {
            zone_id: 3,
            word_id: 42,
        };
        assert_eq!(zwr.zone_id, 3);
        assert_eq!(zwr.word_id, 42);
        let json = serde_json::to_string(&zwr).unwrap();
        let deserialized: ZonedWordRef = serde_json::from_str(&json).unwrap();
        assert_eq!(zwr, deserialized);
    }
}
