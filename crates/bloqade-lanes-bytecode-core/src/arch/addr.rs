#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    Forward = 0,
    Backward = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
    SiteBus = 0,
    WordBus = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocationAddr {
    pub word_id: u32,
    pub site_id: u32,
}

impl LocationAddr {
    // Layout: [word_id:16][site_id:16]
    pub fn encode(&self) -> u32 {
        ((self.word_id as u16 as u32) << 16) | (self.site_id as u16 as u32)
    }

    pub fn decode(bits: u32) -> Self {
        Self {
            word_id: (bits >> 16) & 0xFFFF,
            site_id: bits & 0xFFFF,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneAddr {
    pub direction: Direction,
    pub move_type: MoveType,
    pub word_id: u32,
    pub site_id: u32,
    pub bus_id: u32,
}

impl LaneAddr {
    // Layout across two u32 words:
    //   data0: [word_id:16][site_id:16]
    //   data1: [dir:1][mt:1][pad:14][bus_id:16]
    pub fn encode(&self) -> (u32, u32) {
        let data0 = ((self.word_id as u16 as u32) << 16) | (self.site_id as u16 as u32);
        let data1 = ((self.direction as u32) << 31)
            | ((self.move_type as u32) << 30)
            | (self.bus_id as u16 as u32);
        (data0, data1)
    }

    pub fn encode_u64(&self) -> u64 {
        let (d0, d1) = self.encode();
        (d0 as u64) | ((d1 as u64) << 32)
    }

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

    pub fn decode_u64(bits: u64) -> Self {
        Self::decode(bits as u32, (bits >> 32) as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneAddr {
    pub zone_id: u32,
}

impl ZoneAddr {
    // Layout: [pad:16][zone_id:16]
    pub fn encode(&self) -> u32 {
        self.zone_id as u16 as u32
    }

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
}
