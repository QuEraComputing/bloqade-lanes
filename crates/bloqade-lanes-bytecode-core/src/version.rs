use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Semantic version with major.minor components.
///
/// Packed as `(major << 16) | minor` for binary format compatibility (4 bytes LE).
/// Serialized to JSON as a `"major.minor"` string (e.g. `"1.0"`).
/// Deserializes from:
/// - A string `"major.minor"` (e.g. `"1.0"`) — canonical format.
/// - A small integer N (≤ 65535) — legacy format, interpreted as major=N, minor=0.
/// - A packed integer (> 65535) — legacy format, decoded as `(N >> 16, N & 0xFFFF)`.
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

/// Deserializes from either:
/// - A string `"major.minor"` (e.g. `"1.0"`) — the canonical format.
/// - A small integer N (≤ 65535) — legacy format, major=N, minor=0.
/// - A packed integer (> 65535, ≤ u32::MAX) — legacy format, decoded as `(N >> 16, N & 0xFFFF)`.
impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, Visitor};

        struct VersionVisitor;

        impl<'de> Visitor<'de> for VersionVisitor {
            type Value = Version;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a version string like \"1.0\" or a legacy integer")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Version, E> {
                let (major, minor) = v.split_once('.').ok_or_else(|| {
                    de::Error::custom(format!(
                        "invalid version string '{}': expected 'major.minor'",
                        v
                    ))
                })?;
                let major: u16 = major.parse().map_err(de::Error::custom)?;
                let minor: u16 = minor.parse().map_err(de::Error::custom)?;
                Ok(Version::new(major, minor))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Version, E> {
                if v > u32::MAX as u64 {
                    Err(de::Error::custom(format!(
                        "legacy version integer {} exceeds maximum {}",
                        v,
                        u32::MAX
                    )))
                } else if v <= u16::MAX as u64 {
                    // Small integer: treat as major version (minor = 0)
                    Ok(Version::new(v as u16, 0))
                } else {
                    // Packed format: (major << 16) | minor
                    Ok(Version::from(v as u32))
                }
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Version, E> {
                if v < 0 {
                    Err(de::Error::custom(format!(
                        "version must be non-negative, got {}",
                        v
                    )))
                } else {
                    self.visit_u64(v as u64)
                }
            }
        }

        deserializer.deserialize_any(VersionVisitor)
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
    fn test_version_serde_string_with_minor() {
        let deserialized: Version = serde_json::from_str(r#""2.3""#).unwrap();
        assert_eq!(deserialized, Version::new(2, 3));
    }

    #[test]
    fn test_version_serde_legacy_integer() {
        // Legacy JSON "version": 1 deserializes as Version { major: 1, minor: 0 }
        let deserialized: Version = serde_json::from_str("1").unwrap();
        assert_eq!(deserialized, Version::new(1, 0));
    }

    #[test]
    fn test_version_serde_invalid_string() {
        let err = serde_json::from_str::<Version>(r#""bad""#).unwrap_err();
        assert!(err.to_string().contains("expected 'major.minor'"));
    }

    #[test]
    fn test_version_serde_negative_integer() {
        let err = serde_json::from_str::<Version>("-1").unwrap_err();
        assert!(err.to_string().contains("non-negative"));
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
    fn test_version_serde_legacy_packed_integer() {
        // 65536 = (1 << 16) | 0 = Version(1, 0)
        let deserialized: Version = serde_json::from_str("65536").unwrap();
        assert_eq!(deserialized, Version::new(1, 0));

        // 65537 = (1 << 16) | 1 = Version(1, 1)
        let deserialized: Version = serde_json::from_str("65537").unwrap();
        assert_eq!(deserialized, Version::new(1, 1));

        // 131077 = (2 << 16) | 5 = Version(2, 5)
        let deserialized: Version = serde_json::from_str("131077").unwrap();
        assert_eq!(deserialized, Version::new(2, 5));
    }

    #[test]
    fn test_version_serde_legacy_integer_overflow() {
        // u32::MAX + 1 should be rejected
        let err = serde_json::from_str::<Version>("4294967296").unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_version_serde_legacy_integer_zero() {
        let deserialized: Version = serde_json::from_str("0").unwrap();
        assert_eq!(deserialized, Version::new(0, 0));
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
    fn test_version_serde_wrong_type() {
        let err = serde_json::from_str::<Version>("true").unwrap_err();
        assert!(err.to_string().contains("version string"));
    }

    #[test]
    fn test_version_serde_zero_zero() {
        let v = Version::new(0, 0);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, r#""0.0""#);
        let deserialized: Version = serde_json::from_str(&json).unwrap();
        assert_eq!(v, deserialized);
    }
}
