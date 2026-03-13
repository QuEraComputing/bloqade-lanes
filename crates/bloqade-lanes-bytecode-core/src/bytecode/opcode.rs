use super::encode::DecodeError;

/// Device code identifying which hardware subsystem an instruction targets.
///
/// The device code occupies the least significant byte of the packed opcode:
/// `full_opcode = (instruction_code << 8) | device_code`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeviceCode {
    Cpu = 0x00,
    LaneConstants = 0x0F,
    AtomArrangement = 0x10,
    QuantumGate = 0x11,
    Measurement = 0x12,
    Array = 0x13,
    DetectorObservable = 0x14,
}

impl DeviceCode {
    /// Extract the device code from a raw opcode word.
    pub fn from_opcode(opcode: u32) -> Result<Self, DecodeError> {
        let device_byte = (opcode & 0xFF) as u8;
        Self::from_byte(device_byte)
    }

    /// Parse a device code from a single byte.
    pub fn from_byte(byte: u8) -> Result<Self, DecodeError> {
        match byte {
            0x00 => Ok(DeviceCode::Cpu),
            0x0F => Ok(DeviceCode::LaneConstants),
            0x10 => Ok(DeviceCode::AtomArrangement),
            0x11 => Ok(DeviceCode::QuantumGate),
            0x12 => Ok(DeviceCode::Measurement),
            0x13 => Ok(DeviceCode::Array),
            0x14 => Ok(DeviceCode::DetectorObservable),
            _ => Err(DecodeError::UnknownDeviceCode(byte)),
        }
    }

    /// Returns the hex device code value.
    pub fn code(&self) -> u8 {
        *self as u8
    }
}

/// Cpu instruction codes (FLAIR-aligned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CpuInstCode {
    ConstInt = 0x02,
    ConstFloat = 0x03,
    Dup = 0x04,
    Pop = 0x05,
    Swap = 0x06,
    Return = 0x64,
    Halt = 0xFF,
}

/// LaneConstants instruction codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LaneConstInstCode {
    ConstLoc = 0x00,
    ConstLane = 0x01,
    ConstZone = 0x02,
}

/// AtomArrangement instruction codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AtomArrangementInstCode {
    InitialFill = 0x00,
    Fill = 0x01,
    Move = 0x02,
}

/// QuantumGate instruction codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantumGateInstCode {
    LocalR = 0x00,
    LocalRz = 0x01,
    GlobalR = 0x02,
    GlobalRz = 0x03,
    CZ = 0x04,
}

/// Measurement instruction codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MeasurementInstCode {
    Measure = 0x00,
    AwaitMeasure = 0x01,
}

/// Array instruction codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArrayInstCode {
    NewArray = 0x00,
    GetItem = 0x01,
}

/// DetectorObservable instruction codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DetectorObservableInstCode {
    SetDetector = 0x00,
    SetObservable = 0x01,
}

/// Pack a device code and instruction code into a full opcode.
pub const fn pack_opcode(device: u8, inst: u8) -> u16 {
    ((inst as u16) << 8) | (device as u16)
}

/// Categorized opcode after decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCodeCategory {
    Cpu(CpuInstCode),
    LaneConstants(LaneConstInstCode),
    AtomArrangement(AtomArrangementInstCode),
    QuantumGate(QuantumGateInstCode),
    Measurement(MeasurementInstCode),
    Array(ArrayInstCode),
    DetectorObservable(DetectorObservableInstCode),
}

/// Decode a u32 opcode word into an OpCodeCategory.
pub fn decode_opcode(word: u32) -> Result<OpCodeCategory, DecodeError> {
    let device_byte = (word & 0xFF) as u8;
    let inst_byte = ((word >> 8) & 0xFF) as u8;

    // Verify upper 16 bits are zero
    if word & 0xFFFF_0000 != 0 {
        return Err(DecodeError::UnknownOpcode(word as u16));
    }

    let device = DeviceCode::from_byte(device_byte)?;

    match device {
        DeviceCode::Cpu => match inst_byte {
            0x02 => Ok(OpCodeCategory::Cpu(CpuInstCode::ConstInt)),
            0x03 => Ok(OpCodeCategory::Cpu(CpuInstCode::ConstFloat)),
            0x04 => Ok(OpCodeCategory::Cpu(CpuInstCode::Dup)),
            0x05 => Ok(OpCodeCategory::Cpu(CpuInstCode::Pop)),
            0x06 => Ok(OpCodeCategory::Cpu(CpuInstCode::Swap)),
            0x64 => Ok(OpCodeCategory::Cpu(CpuInstCode::Return)),
            0xFF => Ok(OpCodeCategory::Cpu(CpuInstCode::Halt)),
            _ => Err(DecodeError::UnknownOpcode(word as u16)),
        },
        DeviceCode::LaneConstants => match inst_byte {
            0x00 => Ok(OpCodeCategory::LaneConstants(LaneConstInstCode::ConstLoc)),
            0x01 => Ok(OpCodeCategory::LaneConstants(LaneConstInstCode::ConstLane)),
            0x02 => Ok(OpCodeCategory::LaneConstants(LaneConstInstCode::ConstZone)),
            _ => Err(DecodeError::UnknownOpcode(word as u16)),
        },
        DeviceCode::AtomArrangement => match inst_byte {
            0x00 => Ok(OpCodeCategory::AtomArrangement(
                AtomArrangementInstCode::InitialFill,
            )),
            0x01 => Ok(OpCodeCategory::AtomArrangement(
                AtomArrangementInstCode::Fill,
            )),
            0x02 => Ok(OpCodeCategory::AtomArrangement(
                AtomArrangementInstCode::Move,
            )),
            _ => Err(DecodeError::UnknownOpcode(word as u16)),
        },
        DeviceCode::QuantumGate => match inst_byte {
            0x00 => Ok(OpCodeCategory::QuantumGate(QuantumGateInstCode::LocalR)),
            0x01 => Ok(OpCodeCategory::QuantumGate(QuantumGateInstCode::LocalRz)),
            0x02 => Ok(OpCodeCategory::QuantumGate(QuantumGateInstCode::GlobalR)),
            0x03 => Ok(OpCodeCategory::QuantumGate(QuantumGateInstCode::GlobalRz)),
            0x04 => Ok(OpCodeCategory::QuantumGate(QuantumGateInstCode::CZ)),
            _ => Err(DecodeError::UnknownOpcode(word as u16)),
        },
        DeviceCode::Measurement => match inst_byte {
            0x00 => Ok(OpCodeCategory::Measurement(MeasurementInstCode::Measure)),
            0x01 => Ok(OpCodeCategory::Measurement(
                MeasurementInstCode::AwaitMeasure,
            )),
            _ => Err(DecodeError::UnknownOpcode(word as u16)),
        },
        DeviceCode::Array => match inst_byte {
            0x00 => Ok(OpCodeCategory::Array(ArrayInstCode::NewArray)),
            0x01 => Ok(OpCodeCategory::Array(ArrayInstCode::GetItem)),
            _ => Err(DecodeError::UnknownOpcode(word as u16)),
        },
        DeviceCode::DetectorObservable => match inst_byte {
            0x00 => Ok(OpCodeCategory::DetectorObservable(
                DetectorObservableInstCode::SetDetector,
            )),
            0x01 => Ok(OpCodeCategory::DetectorObservable(
                DetectorObservableInstCode::SetObservable,
            )),
            _ => Err(DecodeError::UnknownOpcode(word as u16)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_opcode() {
        // const_int: device 0x00, inst 0x02 -> 0x0200
        assert_eq!(pack_opcode(0x00, 0x02), 0x0200);
        // const_loc: device 0x0F, inst 0x00 -> 0x000F
        assert_eq!(pack_opcode(0x0F, 0x00), 0x000F);
        // initial_fill: device 0x10, inst 0x00 -> 0x0010
        assert_eq!(pack_opcode(0x10, 0x00), 0x0010);
        // local_r: device 0x11, inst 0x00 -> 0x0011
        assert_eq!(pack_opcode(0x11, 0x00), 0x0011);
        // halt: device 0x00, inst 0xFF -> 0xFF00
        assert_eq!(pack_opcode(0x00, 0xFF), 0xFF00);
    }

    #[test]
    fn test_all_opcodes_decode() {
        let cases: Vec<(u32, OpCodeCategory)> = vec![
            (0x0200, OpCodeCategory::Cpu(CpuInstCode::ConstInt)),
            (0x0300, OpCodeCategory::Cpu(CpuInstCode::ConstFloat)),
            (0x0400, OpCodeCategory::Cpu(CpuInstCode::Dup)),
            (0x0500, OpCodeCategory::Cpu(CpuInstCode::Pop)),
            (0x0600, OpCodeCategory::Cpu(CpuInstCode::Swap)),
            (0x6400, OpCodeCategory::Cpu(CpuInstCode::Return)),
            (0xFF00, OpCodeCategory::Cpu(CpuInstCode::Halt)),
            (
                0x000F,
                OpCodeCategory::LaneConstants(LaneConstInstCode::ConstLoc),
            ),
            (
                0x010F,
                OpCodeCategory::LaneConstants(LaneConstInstCode::ConstLane),
            ),
            (
                0x020F,
                OpCodeCategory::LaneConstants(LaneConstInstCode::ConstZone),
            ),
            (
                0x0010,
                OpCodeCategory::AtomArrangement(AtomArrangementInstCode::InitialFill),
            ),
            (
                0x0110,
                OpCodeCategory::AtomArrangement(AtomArrangementInstCode::Fill),
            ),
            (
                0x0210,
                OpCodeCategory::AtomArrangement(AtomArrangementInstCode::Move),
            ),
            (
                0x0011,
                OpCodeCategory::QuantumGate(QuantumGateInstCode::LocalR),
            ),
            (
                0x0111,
                OpCodeCategory::QuantumGate(QuantumGateInstCode::LocalRz),
            ),
            (
                0x0211,
                OpCodeCategory::QuantumGate(QuantumGateInstCode::GlobalR),
            ),
            (
                0x0311,
                OpCodeCategory::QuantumGate(QuantumGateInstCode::GlobalRz),
            ),
            (0x0411, OpCodeCategory::QuantumGate(QuantumGateInstCode::CZ)),
            (
                0x0012,
                OpCodeCategory::Measurement(MeasurementInstCode::Measure),
            ),
            (
                0x0112,
                OpCodeCategory::Measurement(MeasurementInstCode::AwaitMeasure),
            ),
            (0x0013, OpCodeCategory::Array(ArrayInstCode::NewArray)),
            (0x0113, OpCodeCategory::Array(ArrayInstCode::GetItem)),
            (
                0x0014,
                OpCodeCategory::DetectorObservable(DetectorObservableInstCode::SetDetector),
            ),
            (
                0x0114,
                OpCodeCategory::DetectorObservable(DetectorObservableInstCode::SetObservable),
            ),
        ];

        for (opcode, expected) in cases {
            assert_eq!(
                decode_opcode(opcode).unwrap(),
                expected,
                "failed for 0x{:04x}",
                opcode
            );
        }
    }

    #[test]
    fn test_unknown_opcode() {
        // Unknown device code
        assert!(decode_opcode(0x0001).is_err()); // device 0x01 is reserved
        // Unknown instruction within known device
        assert!(decode_opcode(0x0100).is_err()); // Cpu device, inst 0x01 not assigned
        // Upper bits set
        assert!(decode_opcode(0x00010200).is_err());
    }

    #[test]
    fn test_device_code_from_opcode() {
        assert_eq!(DeviceCode::from_opcode(0x0200).unwrap(), DeviceCode::Cpu);
        assert_eq!(
            DeviceCode::from_opcode(0x000F).unwrap(),
            DeviceCode::LaneConstants
        );
        assert_eq!(
            DeviceCode::from_opcode(0x0010).unwrap(),
            DeviceCode::AtomArrangement
        );
    }

    #[test]
    fn test_device_code_values() {
        assert_eq!(DeviceCode::Cpu.code(), 0x00);
        assert_eq!(DeviceCode::LaneConstants.code(), 0x0F);
        assert_eq!(DeviceCode::AtomArrangement.code(), 0x10);
        assert_eq!(DeviceCode::QuantumGate.code(), 0x11);
        assert_eq!(DeviceCode::Measurement.code(), 0x12);
        assert_eq!(DeviceCode::Array.code(), 0x13);
        assert_eq!(DeviceCode::DetectorObservable.code(), 0x14);
    }
}
