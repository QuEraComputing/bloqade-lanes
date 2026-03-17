use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Semantic version with major.minor components.
///
/// Packed as `(major << 16) | minor` for binary format compatibility (4 bytes LE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
}

impl Version {
    pub fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Two versions are compatible if their major versions match.
    pub fn is_compatible(&self, other: &Version) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl From<u32> for Version {
    fn from(v: u32) -> Self {
        Self {
            major: (v >> 16) as u16,
            minor: v as u16,
        }
    }
}

impl From<Version> for u32 {
    fn from(v: Version) -> u32 {
        ((v.major as u32) << 16) | (v.minor as u32)
    }
}

/// Custom serde: serializes as a u32 using the packed `(major << 16) | minor` format.
/// On deserialization, if the value fits in u16 (high 16 bits are zero), it is
/// treated as a legacy format where the integer is the major version (minor = 0).
/// Otherwise, the packed format is used.
impl Serialize for Version {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let packed: u32 = (*self).into();
        packed.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = u32::deserialize(deserializer)?;
        // Legacy format: small integers like 1, 2, 3 mean major=N, minor=0
        if v <= u16::MAX as u32 {
            Ok(Version::new(v as u16, 0))
        } else {
            Ok(Version::from(v))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_display() {
        assert_eq!(Version::new(1, 0).to_string(), "1.0");
        assert_eq!(Version::new(2, 3).to_string(), "2.3");
    }

    #[test]
    fn test_version_u32_round_trip() {
        let v = Version::new(1, 0);
        let packed: u32 = v.into();
        assert_eq!(packed, 0x00010000);
        assert_eq!(Version::from(packed), v);

        let v2 = Version::new(2, 5);
        let packed2: u32 = v2.into();
        assert_eq!(packed2, 0x00020005);
        assert_eq!(Version::from(packed2), v2);
    }

    #[test]
    fn test_version_from_u32_packed() {
        // From raw packed format: 1 → major=0, minor=1
        let v = Version::from(1u32);
        assert_eq!(v, Version::new(0, 1));
    }

    #[test]
    fn test_version_is_compatible() {
        let v1 = Version::new(1, 0);
        let v1_1 = Version::new(1, 1);
        let v2 = Version::new(2, 0);

        assert!(v1.is_compatible(&v1_1));
        assert!(v1_1.is_compatible(&v1));
        assert!(!v1.is_compatible(&v2));
    }

    #[test]
    fn test_version_serde_round_trip() {
        let v = Version::new(1, 0);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "65536"); // 0x00010000
        let deserialized: Version = serde_json::from_str(&json).unwrap();
        assert_eq!(v, deserialized);
    }

    #[test]
    fn test_version_serde_legacy_format() {
        // Legacy JSON "version": 1 deserializes as Version { major: 1, minor: 0 }
        let deserialized: Version = serde_json::from_str("1").unwrap();
        assert_eq!(deserialized, Version::new(1, 0));
    }
}
