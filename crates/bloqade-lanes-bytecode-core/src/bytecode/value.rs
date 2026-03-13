use super::encode::DecodeError;

// 4-bit type tag constants
pub const TAG_FLOAT: u8 = 0x0;
pub const TAG_INT: u8 = 0x1;
pub const TAG_ARRAY_REF: u8 = 0x2;
pub const TAG_LOCATION: u8 = 0x3;
pub const TAG_LANE: u8 = 0x4;
pub const TAG_ZONE: u8 = 0x5;
pub const TAG_MEASURE_FUTURE: u8 = 0x6;
pub const TAG_DETECTOR_REF: u8 = 0x7;
pub const TAG_OBSERVABLE_REF: u8 = 0x8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CpuValue {
    Float(f64),
    Int(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayValue {
    ArrayRef(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceValue {
    Location(u32),
    Lane(u64),
    Zone(u32),
    MeasureFuture(u32),
    DetectorRef(u32),
    ObservableRef(u32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Cpu(CpuValue),
    Array(ArrayValue),
    Device(DeviceValue),
}

impl Value {
    pub fn type_tag(&self) -> u8 {
        match self {
            Value::Cpu(CpuValue::Float(_)) => TAG_FLOAT,
            Value::Cpu(CpuValue::Int(_)) => TAG_INT,
            Value::Array(ArrayValue::ArrayRef(_)) => TAG_ARRAY_REF,
            Value::Device(DeviceValue::Location(_)) => TAG_LOCATION,
            Value::Device(DeviceValue::Lane(_)) => TAG_LANE,
            Value::Device(DeviceValue::Zone(_)) => TAG_ZONE,
            Value::Device(DeviceValue::MeasureFuture(_)) => TAG_MEASURE_FUTURE,
            Value::Device(DeviceValue::DetectorRef(_)) => TAG_DETECTOR_REF,
            Value::Device(DeviceValue::ObservableRef(_)) => TAG_OBSERVABLE_REF,
        }
    }

    pub fn raw_bits(&self) -> u64 {
        match self {
            Value::Cpu(CpuValue::Float(f)) => f.to_bits(),
            Value::Cpu(CpuValue::Int(v)) => *v as u64,
            Value::Array(ArrayValue::ArrayRef(v)) => *v as u64,
            Value::Device(DeviceValue::Location(v)) => *v as u64,
            Value::Device(DeviceValue::Lane(v)) => *v,
            Value::Device(DeviceValue::Zone(v)) => *v as u64,
            Value::Device(DeviceValue::MeasureFuture(v)) => *v as u64,
            Value::Device(DeviceValue::DetectorRef(v)) => *v as u64,
            Value::Device(DeviceValue::ObservableRef(v)) => *v as u64,
        }
    }

    pub fn from_tag_and_bits(tag: u8, bits: u64) -> Result<Self, DecodeError> {
        match tag {
            TAG_FLOAT => Ok(Value::Cpu(CpuValue::Float(f64::from_bits(bits)))),
            TAG_INT => Ok(Value::Cpu(CpuValue::Int(bits as i64))),
            TAG_ARRAY_REF => Ok(Value::Array(ArrayValue::ArrayRef(bits as u32))),
            TAG_LOCATION => Ok(Value::Device(DeviceValue::Location(bits as u32))),
            TAG_LANE => Ok(Value::Device(DeviceValue::Lane(bits))),
            TAG_ZONE => Ok(Value::Device(DeviceValue::Zone(bits as u32))),
            TAG_MEASURE_FUTURE => Ok(Value::Device(DeviceValue::MeasureFuture(bits as u32))),
            TAG_DETECTOR_REF => Ok(Value::Device(DeviceValue::DetectorRef(bits as u32))),
            TAG_OBSERVABLE_REF => Ok(Value::Device(DeviceValue::ObservableRef(bits as u32))),
            _ => Err(DecodeError::InvalidOperand {
                opcode: 0,
                message: format!("unknown type tag: 0x{:x}", tag),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_tag_round_trip() {
        let values = vec![
            Value::Cpu(CpuValue::Float(1.5)),
            Value::Cpu(CpuValue::Int(42)),
            Value::Array(ArrayValue::ArrayRef(7)),
            Value::Device(DeviceValue::Location(100)),
            Value::Device(DeviceValue::Lane(200)),
            Value::Device(DeviceValue::Zone(300)),
            Value::Device(DeviceValue::MeasureFuture(400)),
            Value::Device(DeviceValue::DetectorRef(500)),
            Value::Device(DeviceValue::ObservableRef(600)),
        ];

        for val in values {
            let tag = val.type_tag();
            let bits = val.raw_bits();
            let reconstructed = Value::from_tag_and_bits(tag, bits).unwrap();
            assert_eq!(val, reconstructed);
        }
    }

    #[test]
    fn test_raw_bits_float() {
        let v = Value::Cpu(CpuValue::Float(1.0));
        assert_eq!(v.raw_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn test_raw_bits_int() {
        let v = Value::Cpu(CpuValue::Int(-1));
        assert_eq!(v.raw_bits(), 0xFFFF_FFFF_FFFF_FFFF);
    }

    #[test]
    fn test_from_tag_and_bits_each_variant() {
        assert_eq!(
            Value::from_tag_and_bits(TAG_FLOAT, 1.0_f64.to_bits()).unwrap(),
            Value::Cpu(CpuValue::Float(1.0))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_INT, 42).unwrap(),
            Value::Cpu(CpuValue::Int(42))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_ARRAY_REF, 5).unwrap(),
            Value::Array(ArrayValue::ArrayRef(5))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_LOCATION, 10).unwrap(),
            Value::Device(DeviceValue::Location(10))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_LANE, 20).unwrap(),
            Value::Device(DeviceValue::Lane(20))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_ZONE, 30).unwrap(),
            Value::Device(DeviceValue::Zone(30))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_MEASURE_FUTURE, 40).unwrap(),
            Value::Device(DeviceValue::MeasureFuture(40))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_DETECTOR_REF, 50).unwrap(),
            Value::Device(DeviceValue::DetectorRef(50))
        );
        assert_eq!(
            Value::from_tag_and_bits(TAG_OBSERVABLE_REF, 60).unwrap(),
            Value::Device(DeviceValue::ObservableRef(60))
        );
    }

    #[test]
    fn test_unknown_tag_produces_error() {
        assert!(Value::from_tag_and_bits(0xF, 0).is_err());
    }
}
