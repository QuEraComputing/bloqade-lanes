use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr as rs_addr;
use bloqade_lanes_bytecode_core::bytecode::instruction as rs;

use crate::arch_python::{PyDirection, PyMoveType};
use crate::validation::{validate_u16_field, validate_u32_field};

#[pyclass(name = "Instruction", frozen, module = "bloqade.lanes.bytecode")]
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
    fn const_loc(word_id: i64, site_id: i64) -> PyResult<Self> {
        let word_id = validate_u16_field("word_id", word_id)?;
        let site_id = validate_u16_field("site_id", site_id)?;
        let addr = rs_addr::LocationAddr { word_id, site_id };
        Ok(Self {
            inner: rs::Instruction::LaneConst(rs::LaneConstInstruction::ConstLoc(addr.encode())),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (move_type, word_id, site_id, bus_id, direction=PyDirection::Forward))]
    fn const_lane(
        move_type: &PyMoveType,
        word_id: i64,
        site_id: i64,
        bus_id: i64,
        direction: PyDirection,
    ) -> PyResult<Self> {
        let word_id = validate_u16_field("word_id", word_id)?;
        let site_id = validate_u16_field("site_id", site_id)?;
        let bus_id = validate_u16_field("bus_id", bus_id)?;
        let addr = rs_addr::LaneAddr {
            direction: direction.to_rs(),
            move_type: move_type.to_rs(),
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
        let zone_id = validate_u16_field("zone_id", zone_id)?;
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
        let arity = validate_u32_field("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::AtomArrangement(rs::AtomArrangementInstruction::InitialFill {
                arity,
            }),
        })
    }

    #[staticmethod]
    fn fill(arity: i64) -> PyResult<Self> {
        let arity = validate_u32_field("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::AtomArrangement(rs::AtomArrangementInstruction::Fill { arity }),
        })
    }

    #[staticmethod]
    #[pyo3(name = "move_")]
    fn move_instr(arity: i64) -> PyResult<Self> {
        let arity = validate_u32_field("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::AtomArrangement(rs::AtomArrangementInstruction::Move { arity }),
        })
    }

    // ── Gate operations ──

    #[staticmethod]
    fn local_r(arity: i64) -> PyResult<Self> {
        let arity = validate_u32_field("arity", arity)?;
        Ok(Self {
            inner: rs::Instruction::QuantumGate(rs::QuantumGateInstruction::LocalR { arity }),
        })
    }

    #[staticmethod]
    fn local_rz(arity: i64) -> PyResult<Self> {
        let arity = validate_u32_field("arity", arity)?;
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
        let arity = validate_u32_field("arity", arity)?;
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
    fn new_array(type_tag: u8, dim0: u16, dim1: u16) -> Self {
        Self {
            inner: rs::Instruction::Array(rs::ArrayInstruction::NewArray {
                type_tag,
                dim0,
                dim1,
            }),
        }
    }

    #[staticmethod]
    fn get_item(ndims: u16) -> Self {
        Self {
            inner: rs::Instruction::Array(rs::ArrayInstruction::GetItem { ndims }),
        }
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
                    "Instruction.const_loc(word_id={}, site_id={})",
                    addr.word_id, addr.site_id
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
                };
                format!(
                    "Instruction.const_lane(move_type={}, word_id={}, site_id={}, bus_id={}, direction={})",
                    mt, addr.word_id, addr.site_id, addr.bus_id, dir
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
