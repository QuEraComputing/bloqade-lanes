use std::fmt;

/// Error returned when a field value is outside its valid range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldRangeError {
    Negative { name: String, value: i64 },
    Overflow { name: String, value: i64, max: u64 },
}

impl fmt::Display for FieldRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldRangeError::Negative { name, value } => {
                write!(f, "{}={} must be non-negative", name, value)
            }
            FieldRangeError::Overflow { name, value, max } => {
                write!(f, "{}={} exceeds maximum {}", name, value, max)
            }
        }
    }
}

impl From<FieldRangeError> for pyo3::PyErr {
    fn from(err: FieldRangeError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// Trait for unsigned integer types that can be validated from i64.
pub trait ValidateFromI64: Sized + Copy {
    const MAX_VALUE: u64;

    /// Try to convert a non-negative i64 that is within range.
    /// Caller guarantees 0 <= value <= MAX_VALUE.
    fn from_i64_unchecked(value: i64) -> Self;
}

impl ValidateFromI64 for u8 {
    const MAX_VALUE: u64 = u8::MAX as u64;
    fn from_i64_unchecked(value: i64) -> Self {
        value as u8
    }
}

impl ValidateFromI64 for u16 {
    const MAX_VALUE: u64 = u16::MAX as u64;
    fn from_i64_unchecked(value: i64) -> Self {
        value as u16
    }
}

impl ValidateFromI64 for u32 {
    const MAX_VALUE: u64 = u32::MAX as u64;
    fn from_i64_unchecked(value: i64) -> Self {
        value as u32
    }
}

/// Validate that an i64 value fits in the range of `T` (0..=T::MAX_VALUE).
pub fn validate_field<T: ValidateFromI64>(name: &str, value: i64) -> Result<T, FieldRangeError> {
    if value < 0 {
        return Err(FieldRangeError::Negative {
            name: name.to_string(),
            value,
        });
    }
    if value as u64 > T::MAX_VALUE {
        return Err(FieldRangeError::Overflow {
            name: name.to_string(),
            value,
            max: T::MAX_VALUE,
        });
    }
    Ok(T::from_i64_unchecked(value))
}

/// Validate that every element in a vector fits in the range of `T`.
/// Error messages include the field name and index (e.g. "zones[3]=-1").
pub fn validate_vec<T: ValidateFromI64>(
    name: &str,
    values: Vec<i64>,
) -> Result<Vec<T>, FieldRangeError> {
    values
        .into_iter()
        .enumerate()
        .map(|(i, v)| validate_field::<T>(&format!("{name}[{i}]"), v))
        .collect()
}

/// Validate a HashMap with i64 keys, converting each key to `T`.
pub fn validate_i64_key_map<T: ValidateFromI64, V>(
    name: &str,
    map: std::collections::HashMap<i64, V>,
) -> Result<std::collections::HashMap<T, V>, FieldRangeError>
where
    T: Eq + std::hash::Hash,
{
    map.into_iter()
        .map(|(k, v)| {
            let k = validate_field::<T>(name, k)?;
            Ok((k, v))
        })
        .collect()
}

/// Validate a HashMap with i64 values, converting each value to `T`.
pub fn validate_i64_value_map<K: Eq + std::hash::Hash, T: ValidateFromI64>(
    name: &str,
    map: std::collections::HashMap<K, i64>,
) -> Result<std::collections::HashMap<K, T>, FieldRangeError> {
    map.into_iter()
        .map(|(k, v)| {
            let v = validate_field::<T>(name, v)?;
            Ok((k, v))
        })
        .collect()
}

/// Validate a HashMap with i64 keys and i64 values, converting both to `T`.
pub fn validate_i64_kv_map<T: ValidateFromI64>(
    key_name: &str,
    value_name: &str,
    map: std::collections::HashMap<i64, i64>,
) -> Result<std::collections::HashMap<T, T>, FieldRangeError>
where
    T: Eq + std::hash::Hash,
{
    map.into_iter()
        .map(|(k, v)| {
            let k = validate_field::<T>(key_name, k)?;
            let v = validate_field::<T>(value_name, v)?;
            Ok((k, v))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    // ── validate_field generic ──

    #[test]
    fn generic_u8() {
        assert_eq!(validate_field::<u8>("x", 0).unwrap(), 0u8);
        assert_eq!(validate_field::<u8>("x", 255).unwrap(), 255u8);
        assert!(validate_field::<u8>("x", 256).is_err());
        assert!(validate_field::<u8>("x", -1).is_err());
    }

    #[test]
    fn generic_u16() {
        assert_eq!(validate_field::<u16>("x", 0).unwrap(), 0u16);
        assert_eq!(validate_field::<u16>("x", 0xFFFF).unwrap(), 0xFFFFu16);
        assert!(validate_field::<u16>("x", 0x10000).is_err());
        assert!(validate_field::<u16>("x", -1).is_err());
    }

    #[test]
    fn generic_u32() {
        assert_eq!(validate_field::<u32>("x", 0).unwrap(), 0u32);
        assert_eq!(
            validate_field::<u32>("x", u32::MAX as i64).unwrap(),
            u32::MAX
        );
        assert!(validate_field::<u32>("x", u32::MAX as i64 + 1).is_err());
        assert!(validate_field::<u32>("x", -1).is_err());
    }

    // ── validate_vec generic ──

    #[test]
    fn generic_vec_u32() {
        assert_eq!(
            validate_vec::<u32>("x", vec![0, 1, 2]).unwrap(),
            vec![0u32, 1, 2]
        );
    }

    #[test]
    fn generic_vec_u16() {
        assert_eq!(
            validate_vec::<u16>("x", vec![0, 100]).unwrap(),
            vec![0u16, 100]
        );
        assert!(validate_vec::<u16>("x", vec![0, 0x10000]).is_err());
    }

    // ── validate_vec edge cases ──

    #[test]
    fn vec_empty() {
        let result: Vec<u32> = validate_vec::<u32>("x", vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn vec_negative_includes_index() {
        let err = validate_vec::<u32>("zones", vec![0, 1, -5]).unwrap_err();
        assert!(matches!(err, FieldRangeError::Negative { .. }));
        assert!(err.to_string().contains("zones[2]=-5"), "got: {err}");
    }

    #[test]
    fn vec_overflow_includes_index() {
        let err = validate_vec::<u32>("ids", vec![0, u32::MAX as i64 + 1]).unwrap_err();
        assert!(matches!(err, FieldRangeError::Overflow { .. }));
        assert!(err.to_string().contains("ids[1]"), "got: {err}");
    }

    #[test]
    fn vec_short_circuits_on_first_error() {
        let err = validate_vec::<u32>("x", vec![-1, -2, -3]).unwrap_err();
        assert!(err.to_string().contains("x[0]=-1"), "got: {err}");
    }

    // ── validate_i64_key_map ──

    #[test]
    fn key_map_valid() {
        let map = HashMap::from([(0i64, "a"), (1, "b")]);
        let result = validate_i64_key_map::<u32, _>("id", map).unwrap();
        assert_eq!(result[&0u32], "a");
        assert_eq!(result[&1u32], "b");
    }

    #[test]
    fn key_map_negative_key() {
        let map = HashMap::from([(-1i64, "a")]);
        let err = validate_i64_key_map::<u32, _>("qubit_id", map).unwrap_err();
        assert!(err.to_string().contains("qubit_id=-1"));
    }

    #[test]
    fn key_map_overflow_key() {
        let map = HashMap::from([(u32::MAX as i64 + 1, "a")]);
        let err = validate_i64_key_map::<u32, _>("qubit_id", map).unwrap_err();
        assert!(matches!(err, FieldRangeError::Overflow { .. }));
    }

    #[test]
    fn key_map_empty() {
        let map: HashMap<i64, &str> = HashMap::new();
        let result = validate_i64_key_map::<u32, _>("id", map).unwrap();
        assert!(result.is_empty());
    }

    // ── validate_i64_value_map ──

    #[test]
    fn value_map_valid() {
        let map = HashMap::from([("a", 0i64), ("b", 255)]);
        let result = validate_i64_value_map::<_, u8>("val", map).unwrap();
        assert_eq!(result["a"], 0u8);
        assert_eq!(result["b"], 255u8);
    }

    #[test]
    fn value_map_negative_value() {
        let map = HashMap::from([("a", -1i64)]);
        let err = validate_i64_value_map::<_, u32>("count", map).unwrap_err();
        assert!(err.to_string().contains("count=-1"));
    }

    #[test]
    fn value_map_overflow_value() {
        let map = HashMap::from([("a", 256i64)]);
        let err = validate_i64_value_map::<_, u8>("val", map).unwrap_err();
        assert!(matches!(err, FieldRangeError::Overflow { .. }));
    }

    // ── validate_i64_kv_map ──

    #[test]
    fn kv_map_valid() {
        let map = HashMap::from([(0i64, 10i64), (1, 20)]);
        let result = validate_i64_kv_map::<u32>("k", "v", map).unwrap();
        assert_eq!(result[&0u32], 10u32);
        assert_eq!(result[&1u32], 20u32);
    }

    #[test]
    fn kv_map_negative_key() {
        let map = HashMap::from([(-1i64, 0i64)]);
        let err = validate_i64_kv_map::<u32>("qubit", "count", map).unwrap_err();
        assert!(err.to_string().contains("qubit=-1"));
    }

    #[test]
    fn kv_map_negative_value() {
        let map = HashMap::from([(0i64, -1i64)]);
        let err = validate_i64_kv_map::<u32>("qubit", "count", map).unwrap_err();
        assert!(err.to_string().contains("count=-1"));
    }

    #[test]
    fn kv_map_empty() {
        let map: HashMap<i64, i64> = HashMap::new();
        let result = validate_i64_kv_map::<u32>("k", "v", map).unwrap();
        assert!(result.is_empty());
    }

    // ── error messages ──

    #[test]
    fn error_display_negative() {
        let err = FieldRangeError::Negative {
            name: "foo".into(),
            value: -42,
        };
        assert_eq!(err.to_string(), "foo=-42 must be non-negative");
    }

    #[test]
    fn error_display_overflow() {
        let err = FieldRangeError::Overflow {
            name: "bar".into(),
            value: 999,
            max: 255,
        };
        assert_eq!(err.to_string(), "bar=999 exceeds maximum 255");
    }
}
