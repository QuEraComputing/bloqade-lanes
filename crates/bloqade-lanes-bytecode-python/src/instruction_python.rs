use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr as rs_addr;
use bloqade_lanes_bytecode_core::bytecode::instruction as rs;

use crate::arch_python::{PyDirection, PyLaneAddr, PyLocationAddr, PyMoveType, PyZoneAddr};
use crate::validation::validate_field;

#[pyclass(
    name = "Instruction",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyInstruction {
    pub(crate) inner: rs::Instruction,
}

#[pymethods]
impl PyInstruction {
    // ── Constants ──

    #[staticmethod]
    fn const_float(value: f64) -> Self {
        Self {
            inner: rs::Instruction::Cpu(rs::CpuInstruction::ConstFloat(value)),
        }
    }

    #[staticmethod]
    fn const_int(value: i64) -> Self {
        Self {
            inner: rs::Instruction::Cpu(rs::CpuInstruction::ConstInt(value)),
        }
    }

    #[staticmethod]
    fn const_loc(zone_id: i64, word_id: i64, site_id: i64) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
        let word_id = validate_field::<u16>("word_id", word_id)? as u32;
        let site_id = validate_field::<u16>("site_id", site_id)? as u32;
        let addr = rs_addr::LocationAddr {
            zone_id,
            word_id,
            site_id,
        };
        Ok(Self {
            inner: rs::Instruction::LaneConst(rs::LaneConstInstruction::ConstLoc(addr.encode())),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (move_type, zone_id, word_id, site_id, bus_id, direction=PyDirection::Forward))]
    fn const_lane(
        move_type: &PyMoveType,
        zone_id: i64,
        word_id: i64,
        site_id: i64,
        bus_id: i64,
        direction: PyDirection,
    ) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
        let word_id = validate_field::<u16>("word_id", word_id)? as u32;
        let site_id = validate_field::<u16>("site_id", site_id)? as u32;
        let bus_id = validate_field::<u16>("bus_id", bus_id)? as u32;
        let addr = rs_addr::LaneAddr {
            direction: direction.to_rs(),
            move_type: move_type.to_rs(),
            zone_id,
            word_id,
            site_id,
            bus_id,
        };
        let (d0, d1) = addr.encode();
        Ok(Self {
            inner: rs::Instruction::LaneConst(rs::LaneConstInstruction::ConstLane(d0, d1)),
        })
    }

    #[staticmethod]
    fn const_zone(zone_id: i64) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
        let addr = rs_addr::ZoneAddr { zone_id };
        Ok(Self {
            inner: rs::Instruction::LaneConst(rs::LaneConstInstruction::ConstZone(addr.encode())),
        })
    }

    // ── Stack manipulation ──

    #[staticmethod]
    fn pop() -> Self {
        Self {
            inner: rs::Instruction::Cpu(rs::CpuInstruction::Pop),
        }
    }

    #[staticmethod]
    fn dup() -> Self {
        Self {
            inner: rs::Instruction::Cpu(rs::CpuInstruction::Dup),
        }
    }

    #[staticmethod]
    fn swap() -> Self {
        Self {
            inner: rs::Instruction::Cpu(rs::CpuInstruction::Swap),
        }
    }

    // ── Atom operations ──

    #[staticmethod]
    fn initial_fill(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::AtomArrangement(rs::AtomArrangementInstruction::InitialFill {
                arity,
            }),
        })
    }

    #[staticmethod]
    fn fill(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::AtomArrangement(rs::AtomArrangementInstruction::Fill { arity }),
        })
    }

    #[staticmethod]
    #[pyo3(name = "move_")]
    fn move_instr(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::AtomArrangement(rs::AtomArrangementInstruction::Move { arity }),
        })
    }

    // ── Gate operations ──

    #[staticmethod]
    fn local_r(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::QuantumGate(rs::QuantumGateInstruction::LocalR { arity }),
        })
    }

    #[staticmethod]
    fn local_rz(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::QuantumGate(rs::QuantumGateInstruction::LocalRz { arity }),
        })
    }

    #[staticmethod]
    fn global_r() -> Self {
        Self {
            inner: rs::Instruction::QuantumGate(rs::QuantumGateInstruction::GlobalR),
        }
    }

    #[staticmethod]
    fn global_rz() -> Self {
        Self {
            inner: rs::Instruction::QuantumGate(rs::QuantumGateInstruction::GlobalRz),
        }
    }

    #[staticmethod]
    fn cz() -> Self {
        Self {
            inner: rs::Instruction::QuantumGate(rs::QuantumGateInstruction::CZ),
        }
    }

    // ── Measurement ──

    #[staticmethod]
    fn measure(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::Measurement(rs::MeasurementInstruction::Measure { arity }),
        })
    }

    #[staticmethod]
    fn await_measure() -> Self {
        Self {
            inner: rs::Instruction::Measurement(rs::MeasurementInstruction::AwaitMeasure),
        }
    }

    // ── Array ──

    #[staticmethod]
    #[pyo3(signature = (type_tag, dim0, dim1=0))]
    fn new_array(type_tag: i64, dim0: i64, dim1: i64) -> PyResult<Self> {
        let type_tag = validate_field::<u8>("type_tag", type_tag)?;
        let dim0 = validate_field::<u16>("dim0", dim0)?;
        let dim1 = validate_field::<u16>("dim1", dim1)?;
        Ok(Self {
            inner: rs::Instruction::Array(rs::ArrayInstruction::NewArray {
                type_tag,
                dim0,
                dim1,
            }),
        })
    }

    #[staticmethod]
    fn get_item(ndims: i64) -> PyResult<Self> {
        let ndims = validate_field::<u16>("ndims", ndims)?;
        Ok(Self {
            inner: rs::Instruction::Array(rs::ArrayInstruction::GetItem { ndims }),
        })
    }

    // ── Detector / Observable ──

    #[staticmethod]
    fn set_detector() -> Self {
        Self {
            inner: rs::Instruction::DetectorObservable(
                rs::DetectorObservableInstruction::SetDetector,
            ),
        }
    }

    #[staticmethod]
    fn set_observable() -> Self {
        Self {
            inner: rs::Instruction::DetectorObservable(
                rs::DetectorObservableInstruction::SetObservable,
            ),
        }
    }

    // ── Control ──

    #[staticmethod]
    #[pyo3(name = "return_")]
    fn return_instr() -> Self {
        Self {
            inner: rs::Instruction::Cpu(rs::CpuInstruction::Return),
        }
    }

    #[staticmethod]
    fn halt() -> Self {
        Self {
            inner: rs::Instruction::Cpu(rs::CpuInstruction::Halt),
        }
    }

    // ── Introspection ──

    #[getter]
    fn opcode(&self) -> u16 {
        self.inner.opcode()
    }

    fn op_name(&self) -> &'static str {
        match &self.inner {
            rs::Instruction::Cpu(cpu) => match cpu {
                rs::CpuInstruction::ConstFloat(_) => "const_float",
                rs::CpuInstruction::ConstInt(_) => "const_int",
                rs::CpuInstruction::Pop => "pop",
                rs::CpuInstruction::Dup => "dup",
                rs::CpuInstruction::Swap => "swap",
                rs::CpuInstruction::Return => "return_",
                rs::CpuInstruction::Halt => "halt",
            },
            rs::Instruction::LaneConst(lc) => match lc {
                rs::LaneConstInstruction::ConstLoc(_) => "const_loc",
                rs::LaneConstInstruction::ConstLane(_, _) => "const_lane",
                rs::LaneConstInstruction::ConstZone(_) => "const_zone",
            },
            rs::Instruction::AtomArrangement(aa) => match aa {
                rs::AtomArrangementInstruction::InitialFill { .. } => "initial_fill",
                rs::AtomArrangementInstruction::Fill { .. } => "fill",
                rs::AtomArrangementInstruction::Move { .. } => "move_",
            },
            rs::Instruction::QuantumGate(qg) => match qg {
                rs::QuantumGateInstruction::LocalR { .. } => "local_r",
                rs::QuantumGateInstruction::LocalRz { .. } => "local_rz",
                rs::QuantumGateInstruction::GlobalR => "global_r",
                rs::QuantumGateInstruction::GlobalRz => "global_rz",
                rs::QuantumGateInstruction::CZ => "cz",
            },
            rs::Instruction::Measurement(m) => match m {
                rs::MeasurementInstruction::Measure { .. } => "measure",
                rs::MeasurementInstruction::AwaitMeasure => "await_measure",
            },
            rs::Instruction::Array(arr) => match arr {
                rs::ArrayInstruction::NewArray { .. } => "new_array",
                rs::ArrayInstruction::GetItem { .. } => "get_item",
            },
            rs::Instruction::DetectorObservable(dob) => match dob {
                rs::DetectorObservableInstruction::SetDetector => "set_detector",
                rs::DetectorObservableInstruction::SetObservable => "set_observable",
            },
        }
    }

    fn float_value(&self) -> PyResult<f64> {
        match &self.inner {
            rs::Instruction::Cpu(rs::CpuInstruction::ConstFloat(f)) => Ok(*f),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "float_value() is only valid on const_float",
            )),
        }
    }

    fn int_value(&self) -> PyResult<i64> {
        match &self.inner {
            rs::Instruction::Cpu(rs::CpuInstruction::ConstInt(n)) => Ok(*n),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "int_value() is only valid on const_int",
            )),
        }
    }

    fn location_address(&self) -> PyResult<PyLocationAddr> {
        match &self.inner {
            rs::Instruction::LaneConst(rs::LaneConstInstruction::ConstLoc(bits)) => {
                let addr = rs_addr::LocationAddr::decode(*bits);
                Ok(PyLocationAddr { inner: addr })
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "location_address() is only valid on const_loc",
            )),
        }
    }

    fn lane_address(&self) -> PyResult<PyLaneAddr> {
        match &self.inner {
            rs::Instruction::LaneConst(rs::LaneConstInstruction::ConstLane(d0, d1)) => {
                let addr = rs_addr::LaneAddr::decode(*d0, *d1);
                Ok(PyLaneAddr { inner: addr })
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "lane_address() is only valid on const_lane",
            )),
        }
    }

    fn zone_address(&self) -> PyResult<PyZoneAddr> {
        match &self.inner {
            rs::Instruction::LaneConst(rs::LaneConstInstruction::ConstZone(bits)) => {
                let addr = rs_addr::ZoneAddr::decode(*bits);
                Ok(PyZoneAddr { inner: addr })
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "zone_address() is only valid on const_zone",
            )),
        }
    }

    fn arity(&self) -> PyResult<u32> {
        match &self.inner {
            rs::Instruction::AtomArrangement(
                rs::AtomArrangementInstruction::InitialFill { arity }
                | rs::AtomArrangementInstruction::Fill { arity }
                | rs::AtomArrangementInstruction::Move { arity },
            ) => Ok(*arity),
            rs::Instruction::QuantumGate(
                rs::QuantumGateInstruction::LocalR { arity }
                | rs::QuantumGateInstruction::LocalRz { arity },
            ) => Ok(*arity),
            rs::Instruction::Measurement(rs::MeasurementInstruction::Measure { arity }) => {
                Ok(*arity)
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "arity() not applicable to this opcode",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format_instruction(&self.inner)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

fn format_instruction(instr: &rs::Instruction) -> String {
    match instr {
        rs::Instruction::Cpu(cpu) => match cpu {
            rs::CpuInstruction::ConstFloat(f) => format!("Instruction.const_float({})", f),
            rs::CpuInstruction::ConstInt(n) => format!("Instruction.const_int({})", n),
            rs::CpuInstruction::Pop => "Instruction.pop()".to_string(),
            rs::CpuInstruction::Dup => "Instruction.dup()".to_string(),
            rs::CpuInstruction::Swap => "Instruction.swap()".to_string(),
            rs::CpuInstruction::Return => "Instruction.return_()".to_string(),
            rs::CpuInstruction::Halt => "Instruction.halt()".to_string(),
        },
        rs::Instruction::LaneConst(lc) => match lc {
            rs::LaneConstInstruction::ConstLoc(bits) => {
                let addr = rs_addr::LocationAddr::decode(*bits);
                format!(
                    "Instruction.const_loc(zone_id={}, word_id={}, site_id={})",
                    addr.zone_id, addr.word_id, addr.site_id
                )
            }
            rs::LaneConstInstruction::ConstLane(d0, d1) => {
                let addr = rs_addr::LaneAddr::decode(*d0, *d1);
                let dir = match addr.direction {
                    rs_addr::Direction::Forward => "Direction.FORWARD",
                    rs_addr::Direction::Backward => "Direction.BACKWARD",
                };
                let mt = match addr.move_type {
                    rs_addr::MoveType::SiteBus => "MoveType.SITE",
                    rs_addr::MoveType::WordBus => "MoveType.WORD",
                    rs_addr::MoveType::ZoneBus => "MoveType.ZONE",
                };
                format!(
                    "Instruction.const_lane(move_type={}, zone_id={}, word_id={}, site_id={}, bus_id={}, direction={})",
                    mt, addr.zone_id, addr.word_id, addr.site_id, addr.bus_id, dir
                )
            }
            rs::LaneConstInstruction::ConstZone(bits) => {
                let addr = rs_addr::ZoneAddr::decode(*bits);
                format!("Instruction.const_zone(zone_id={})", addr.zone_id)
            }
        },
        rs::Instruction::AtomArrangement(aa) => match aa {
            rs::AtomArrangementInstruction::InitialFill { arity } => {
                format!("Instruction.initial_fill({})", arity)
            }
            rs::AtomArrangementInstruction::Fill { arity } => {
                format!("Instruction.fill({})", arity)
            }
            rs::AtomArrangementInstruction::Move { arity } => {
                format!("Instruction.move_({})", arity)
            }
        },
        rs::Instruction::QuantumGate(qg) => match qg {
            rs::QuantumGateInstruction::LocalR { arity } => {
                format!("Instruction.local_r({})", arity)
            }
            rs::QuantumGateInstruction::LocalRz { arity } => {
                format!("Instruction.local_rz({})", arity)
            }
            rs::QuantumGateInstruction::GlobalR => "Instruction.global_r()".to_string(),
            rs::QuantumGateInstruction::GlobalRz => "Instruction.global_rz()".to_string(),
            rs::QuantumGateInstruction::CZ => "Instruction.cz()".to_string(),
        },
        rs::Instruction::Measurement(m) => match m {
            rs::MeasurementInstruction::Measure { arity } => {
                format!("Instruction.measure({})", arity)
            }
            rs::MeasurementInstruction::AwaitMeasure => "Instruction.await_measure()".to_string(),
        },
        rs::Instruction::Array(arr) => match arr {
            rs::ArrayInstruction::NewArray {
                type_tag,
                dim0,
                dim1,
            } => {
                if *dim1 == 0 {
                    format!("Instruction.new_array({}, {})", type_tag, dim0)
                } else {
                    format!("Instruction.new_array({}, {}, {})", type_tag, dim0, dim1)
                }
            }
            rs::ArrayInstruction::GetItem { ndims } => {
                format!("Instruction.get_item({})", ndims)
            }
        },
        rs::Instruction::DetectorObservable(dob) => match dob {
            rs::DetectorObservableInstruction::SetDetector => {
                "Instruction.set_detector()".to_string()
            }
            rs::DetectorObservableInstruction::SetObservable => {
                "Instruction.set_observable()".to_string()
            }
        },
    }
}
