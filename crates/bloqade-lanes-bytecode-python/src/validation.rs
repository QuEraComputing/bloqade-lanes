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

/// Validate that a field value fits in 16 bits (0..=65535).
pub fn validate_u16_field(name: &str, value: i64) -> Result<u32, FieldRangeError> {
    if value < 0 {
        return Err(FieldRangeError::Negative {
            name: name.to_string(),
            value,
        });
    }
    if value > 0xFFFF {
        return Err(FieldRangeError::Overflow {
            name: name.to_string(),
            value,
            max: 0xFFFF,
        });
    }
    Ok(value as u32)
}

/// Validate that a field value fits in 32 bits (0..=4294967295).
pub fn validate_u32_field(name: &str, value: i64) -> Result<u32, FieldRangeError> {
    if value < 0 {
        return Err(FieldRangeError::Negative {
            name: name.to_string(),
            value,
        });
    }
    if value > u32::MAX as i64 {
        return Err(FieldRangeError::Overflow {
            name: name.to_string(),
            value,
            max: u32::MAX as u64,
        });
    }
    Ok(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_valid_range() {
        assert_eq!(validate_u16_field("x", 0).unwrap(), 0);
        assert_eq!(validate_u16_field("x", 42).unwrap(), 42);
        assert_eq!(validate_u16_field("x", 0xFFFF).unwrap(), 0xFFFF);
    }

    #[test]
    fn u16_negative() {
        let err = validate_u16_field("foo", -1).unwrap_err();
        assert!(matches!(err, FieldRangeError::Negative { .. }));
        assert!(err.to_string().contains("foo=-1"));
    }

    #[test]
    fn u16_overflow() {
        let err = validate_u16_field("bar", 0x10000).unwrap_err();
        assert!(matches!(err, FieldRangeError::Overflow { .. }));
        assert!(err.to_string().contains("bar=65536"));
    }

    #[test]
    fn u16_large_negative() {
        assert!(validate_u16_field("x", i64::MIN).is_err());
    }

    #[test]
    fn u32_valid_range() {
        assert_eq!(validate_u32_field("x", 0).unwrap(), 0);
        assert_eq!(validate_u32_field("x", u32::MAX as i64).unwrap(), u32::MAX);
    }

    #[test]
    fn u32_negative() {
        let err = validate_u32_field("arity", -1).unwrap_err();
        assert!(matches!(err, FieldRangeError::Negative { .. }));
        assert!(err.to_string().contains("arity=-1"));
    }

    #[test]
    fn u32_overflow() {
        let err = validate_u32_field("n", u32::MAX as i64 + 1).unwrap_err();
        assert!(matches!(err, FieldRangeError::Overflow { .. }));
        assert!(err.to_string().contains("exceeds maximum"));
    }
}
