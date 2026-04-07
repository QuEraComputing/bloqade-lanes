//! Bit-packed address types for bytecode instructions.
//!
//! These types encode device-level addresses into compact integer
//! representations used in the 16-byte instruction format.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
    /// Moves atoms between zones (value 2).
    ZoneBus = 2,
}

/// Bit-packed atom location address (zone + word + site).
///
/// Encodes `zone_id` (8 bits), `word_id` (16 bits), and `site_id` (16 bits)
/// into a 64-bit word.
///
/// Layout: `[zone_id:8][word_id:16][site_id:16][pad:24]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocationAddr {
    pub zone_id: u32,
    pub word_id: u32,
    pub site_id: u32,
}

impl LocationAddr {
    /// Encode to a 64-bit packed integer.
    ///
    /// Layout: `[zone_id:8][word_id:16][site_id:16][pad:24]`
    pub fn encode(&self) -> u64 {
        ((self.zone_id as u8 as u64) << 56)
            | ((self.word_id as u16 as u64) << 40)
            | ((self.site_id as u16 as u64) << 24)
    }

    /// Decode a 64-bit packed integer into a `LocationAddr`.
    pub fn decode(bits: u64) -> Self {
        Self {
            zone_id: ((bits >> 56) & 0xFF) as u32,
            word_id: ((bits >> 40) & 0xFFFF) as u32,
            site_id: ((bits >> 24) & 0xFFFF) as u32,
        }
    }
}

impl Serialize for LocationAddr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.encode())
    }
}

impl<'de> Deserialize<'de> for LocationAddr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits = u64::deserialize(deserializer)?;
        Ok(Self::decode(bits))
    }
}

/// Bit-packed lane address for atom move operations.
///
/// Encodes direction (1 bit), move type (2 bits), zone_id (8 bits),
/// word_id (16 bits), site_id (16 bits), and bus_id (16 bits) across
/// two 32-bit data words.
///
/// Layout:
/// - data0: `[word_id:16][site_id:16]`
/// - data1: `[dir:1][mt:2][zone_id:8][pad:5][bus_id:16]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneAddr {
    pub direction: Direction,
    pub move_type: MoveType,
    pub zone_id: u32,
    pub word_id: u32,
    pub site_id: u32,
    pub bus_id: u32,
}

impl LaneAddr {
    /// Encode to two 32-bit data words `(data0, data1)`.
    pub fn encode(&self) -> (u32, u32) {
        let data0 = ((self.word_id as u16 as u32) << 16) | (self.site_id as u16 as u32);
        let data1 = ((self.direction as u32) << 31)
            | ((self.move_type as u32) << 29)
            | ((self.zone_id as u8 as u32) << 21)
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
        let mt_bits = (data1 >> 29) & 0x3;
        let move_type = match mt_bits {
            0 => MoveType::SiteBus,
            1 => MoveType::WordBus,
            2 => MoveType::ZoneBus,
            _ => panic!("invalid move type bits: {}", mt_bits),
        };
        Self {
            direction,
            move_type,
            zone_id: ((data1 >> 21) & 0xFF) as u32,
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
    fn test_move_type_zone_bus() {
        assert_eq!(MoveType::SiteBus as u8, 0);
        assert_eq!(MoveType::WordBus as u8, 1);
        assert_eq!(MoveType::ZoneBus as u8, 2);
    }

    #[test]
    fn test_location_addr_64bit_round_trip() {
        let addr = LocationAddr {
            zone_id: 5,
            word_id: 0x1234,
            site_id: 0x5678,
        };
        let bits = addr.encode();
        assert_eq!(LocationAddr::decode(bits), addr);
        assert_eq!((bits >> 56) & 0xFF, 5);
        assert_eq!((bits >> 40) & 0xFFFF, 0x1234);
        assert_eq!((bits >> 24) & 0xFFFF, 0x5678);
        assert_eq!(bits & 0xFFFFFF, 0);
    }

    #[test]
    fn test_location_addr_zero() {
        let addr = LocationAddr {
            zone_id: 0,
            word_id: 0,
            site_id: 0,
        };
        assert_eq!(addr.encode(), 0u64);
        assert_eq!(LocationAddr::decode(0), addr);
    }

    #[test]
    fn test_lane_addr_round_trip() {
        let addr = LaneAddr {
            direction: Direction::Backward,
            move_type: MoveType::WordBus,
            zone_id: 0,
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
        assert_eq!((data1 >> 29) & 0x3, 1); // move_type = WordBus
        assert_eq!(data1 & 0xFFFF, 0x9ABC); // bus_id
    }

    #[test]
    fn test_lane_addr_forward_sitebus() {
        let addr = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
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
            zone_id: 0,
            word_id: 1,
            site_id: 0,
            bus_id: 0,
        };
        let packed = addr.encode_u64();
        assert_eq!(LaneAddr::decode_u64(packed), addr);
    }

    #[test]
    fn test_lane_addr_with_zone_id() {
        let addr = LaneAddr {
            direction: Direction::Backward,
            move_type: MoveType::ZoneBus,
            zone_id: 7,
            word_id: 0x1234,
            site_id: 0x5678,
            bus_id: 0x9ABC,
        };
        let (data0, data1) = addr.encode();
        let decoded = LaneAddr::decode(data0, data1);
        assert_eq!(decoded, addr);
        assert_eq!((data0 >> 16) & 0xFFFF, 0x1234);
        assert_eq!(data0 & 0xFFFF, 0x5678);
        assert_eq!((data1 >> 31) & 1, 1);
        assert_eq!((data1 >> 29) & 0x3, 2);
        assert_eq!((data1 >> 21) & 0xFF, 7);
        assert_eq!(data1 & 0xFFFF, 0x9ABC);
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
