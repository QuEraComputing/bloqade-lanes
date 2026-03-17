use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Semantic version with major.minor components.
///
/// Packed as `(major << 16) | minor` for binary format compatibility (4 bytes LE).
/// Serialized to/from JSON as a `"major.minor"` string (e.g. `"1.0"`).
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

/// Serializes as a `"major.minor"` string (e.g. `"1.0"`).
impl Serialize for Version {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

/// Deserializes from a `"major.minor"` string (e.g. `"1.0"`).
impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let (major, minor) = s.split_once('.').ok_or_else(|| {
            serde::de::Error::custom(format!(
                "invalid version string '{}': expected 'major.minor'",
                s
            ))
        })?;
        let major: u16 = major.parse().map_err(serde::de::Error::custom)?;
        let minor: u16 = minor.parse().map_err(serde::de::Error::custom)?;
        Ok(Version::new(major, minor))
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
        assert_eq!(json, r#""1.0""#);
        let deserialized: Version = serde_json::from_str(&json).unwrap();
        assert_eq!(v, deserialized);
    }

    #[test]
    fn test_version_serde_round_trip_with_minor() {
        let v = Version::new(1, 2);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, r#""1.2""#);
        let deserialized: Version = serde_json::from_str(&json).unwrap();
        assert_eq!(v, deserialized);
    }

    #[test]
    fn test_version_serde_zero_zero() {
        let v = Version::new(0, 0);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, r#""0.0""#);
        let deserialized: Version = serde_json::from_str(&json).unwrap();
        assert_eq!(v, deserialized);
    }

    #[test]
    fn test_version_serde_invalid_no_dot() {
        let err = serde_json::from_str::<Version>(r#""bad""#).unwrap_err();
        assert!(err.to_string().contains("expected 'major.minor'"));
    }

    #[test]
    fn test_version_serde_invalid_major() {
        let err = serde_json::from_str::<Version>(r#""abc.1""#).unwrap_err();
        assert!(err.to_string().contains("invalid digit"));
    }

    #[test]
    fn test_version_serde_invalid_minor() {
        let err = serde_json::from_str::<Version>(r#""1.abc""#).unwrap_err();
        assert!(err.to_string().contains("invalid digit"));
    }

    #[test]
    fn test_version_serde_rejects_integer() {
        let err = serde_json::from_str::<Version>("1").unwrap_err();
        assert!(err.to_string().contains("invalid type: integer"));
    }

    #[test]
    fn test_version_serde_rejects_boolean() {
        let err = serde_json::from_str::<Version>("true").unwrap_err();
        assert!(err.to_string().contains("invalid type: boolean"));
    }
}
