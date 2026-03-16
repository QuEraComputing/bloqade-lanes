use pyo3::prelude::*;

/// Validate that a field value fits in 16 bits (0..=65535).
pub fn validate_u16_field(name: &str, value: i64) -> PyResult<u32> {
    if value < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}={} must be non-negative",
            name, value
        )));
    }
    if value > 0xFFFF {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}={} exceeds maximum 65535",
            name, value
        )));
    }
    Ok(value as u32)
}

/// Validate that a field value fits in 32 bits (0..=4294967295).
pub fn validate_u32_field(name: &str, value: i64) -> PyResult<u32> {
    if value < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}={} must be non-negative",
            name, value
        )));
    }
    if value > u32::MAX as i64 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{}={} exceeds maximum {}",
            name,
            value,
            u32::MAX
        )));
    }
    Ok(value as u32)
}
