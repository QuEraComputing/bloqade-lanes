use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr as rs_addr;
use bloqade_lanes_bytecode_core::isa::Instruction as VInst;
use vihaco::instruction::OpCode;
use vihaco::value::Value;
use vihaco_cpu::Instruction as Cpu;

use crate::arch_python::{PyDirection, PyLaneAddr, PyLocationAddr, PyMoveType, PyZoneAddr};
use crate::validation::validate_field;

#[pyclass(
    name = "Instruction",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyInstruction {
    pub(crate) inner: VInst,
}

#[pymethods]
impl PyInstruction {
    // ── Constants ──
    // CPU const pushes are reused from vihaco-cpu (a typed Value), so the
    // legacy `const_float` / `const_int` factories map onto `Cpu(Const(..))`.

    #[staticmethod]
    fn const_float(value: f64) -> Self {
        Self {
            inner: VInst::Cpu(Cpu::Const(Value::F64(value))),
        }
    }

    #[staticmethod]
    fn const_int(value: i64) -> Self {
        Self {
            inner: VInst::Cpu(Cpu::Const(Value::I64(value))),
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
            inner: VInst::ConstLoc(addr.encode()),
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
        Ok(Self {
            inner: VInst::ConstLane(addr.encode_u64()),
        })
    }

    #[staticmethod]
    fn const_zone(zone_id: i64) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
        let addr = rs_addr::ZoneAddr { zone_id };
        Ok(Self {
            inner: VInst::ConstZone(addr.encode()),
        })
    }

    // ── Stack manipulation ──
    // `pop`/`swap` are lanes-native; `dup` is reused from vihaco-cpu.

    #[staticmethod]
    fn pop() -> Self {
        Self { inner: VInst::Pop }
    }

    #[staticmethod]
    fn dup() -> Self {
        Self {
            inner: VInst::Cpu(Cpu::Dup),
        }
    }

    #[staticmethod]
    fn swap() -> Self {
        Self { inner: VInst::Swap }
    }

    // ── Atom operations ──

    #[staticmethod]
    fn initial_fill(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::InitialFill(arity),
        })
    }

    #[staticmethod]
    fn fill(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Fill(arity),
        })
    }

    #[staticmethod]
    #[pyo3(name = "move_")]
    fn move_instr(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Move(arity),
        })
    }

    // ── Gate operations ──

    #[staticmethod]
    fn local_r(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::LocalR(arity),
        })
    }

    #[staticmethod]
    fn local_rz(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::LocalRz(arity),
        })
    }

    #[staticmethod]
    fn global_r() -> Self {
        Self {
            inner: VInst::GlobalR,
        }
    }

    #[staticmethod]
    fn global_rz() -> Self {
        Self {
            inner: VInst::GlobalRz,
        }
    }

    #[staticmethod]
    fn cz() -> Self {
        Self { inner: VInst::Cz }
    }

    // ── Measurement ──

    #[staticmethod]
    fn measure(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Measure(arity),
        })
    }

    #[staticmethod]
    fn await_measure() -> Self {
        Self {
            inner: VInst::AwaitMeasure,
        }
    }

    // ── Array ──
    // type_tag/dim0/dim1 are validated as in the legacy API (u8/u16) then
    // widened to the u32 operands the vihaco ISA uses.

    #[staticmethod]
    #[pyo3(signature = (type_tag, dim0, dim1=0))]
    fn new_array(type_tag: i64, dim0: i64, dim1: i64) -> PyResult<Self> {
        let type_tag = validate_field::<u8>("type_tag", type_tag)? as u32;
        let dim0 = validate_field::<u16>("dim0", dim0)? as u32;
        let dim1 = validate_field::<u16>("dim1", dim1)? as u32;
        Ok(Self {
            inner: VInst::NewArray(type_tag, dim0, dim1),
        })
    }

    #[staticmethod]
    fn get_item(ndims: i64) -> PyResult<Self> {
        let ndims = validate_field::<u16>("ndims", ndims)? as u32;
        Ok(Self {
            inner: VInst::GetItem(ndims),
        })
    }

    // ── Detector / Observable ──

    #[staticmethod]
    fn set_detector() -> Self {
        Self {
            inner: VInst::SetDetector,
        }
    }

    #[staticmethod]
    fn set_observable() -> Self {
        Self {
            inner: VInst::SetObservable,
        }
    }

    // ── Control ──
    // `return` is lanes-native; `halt` is reused from vihaco-cpu.

    #[staticmethod]
    #[pyo3(name = "return_")]
    fn return_instr() -> Self {
        Self {
            inner: VInst::Return,
        }
    }

    #[staticmethod]
    fn halt() -> Self {
        Self {
            inner: VInst::Cpu(Cpu::Halt),
        }
    }

    // ── Introspection ──

    /// The vihaco opcode byte for this instruction (the outer enum's opcode;
    /// for `Cpu(..)` this is the CPU device's opcode, not the nested one).
    #[getter]
    fn opcode(&self) -> u16 {
        OpCode::opcode(&self.inner) as u16
    }

    fn op_name(&self) -> &'static str {
        match &self.inner {
            VInst::Pop => "pop",
            VInst::Swap => "swap",
            VInst::Return => "return",
            VInst::ConstLoc(_) => "const_loc",
            VInst::ConstLane(_) => "const_lane",
            VInst::ConstZone(_) => "const_zone",
            VInst::InitialFill(_) => "initial_fill",
            VInst::Fill(_) => "fill",
            VInst::Move(_) => "move",
            VInst::LocalR(_) => "local_r",
            VInst::LocalRz(_) => "local_rz",
            VInst::GlobalR => "global_r",
            VInst::GlobalRz => "global_rz",
            VInst::Cz => "cz",
            VInst::Measure(_) => "measure",
            VInst::AwaitMeasure => "await_measure",
            VInst::NewArray(..) => "new_array",
            VInst::GetItem(_) => "get_item",
            VInst::SetDetector => "set_detector",
            VInst::SetObservable => "set_observable",
            VInst::Cpu(cpu) => cpu_op_name(cpu),
        }
    }

    fn float_value(&self) -> PyResult<f64> {
        match &self.inner {
            VInst::Cpu(Cpu::Const(Value::F64(f))) => Ok(*f),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "float_value() is only valid on const_float",
            )),
        }
    }

    fn int_value(&self) -> PyResult<i64> {
        match &self.inner {
            VInst::Cpu(Cpu::Const(Value::I64(n))) => Ok(*n),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "int_value() is only valid on const_int",
            )),
        }
    }

    fn location_address(&self) -> PyResult<PyLocationAddr> {
        match &self.inner {
            VInst::ConstLoc(bits) => Ok(PyLocationAddr {
                inner: rs_addr::LocationAddr::decode(*bits),
            }),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "location_address() is only valid on const_loc",
            )),
        }
    }

    fn lane_address(&self) -> PyResult<PyLaneAddr> {
        match &self.inner {
            VInst::ConstLane(bits) => Ok(PyLaneAddr {
                inner: rs_addr::LaneAddr::decode_u64(*bits),
            }),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "lane_address() is only valid on const_lane",
            )),
        }
    }

    fn zone_address(&self) -> PyResult<PyZoneAddr> {
        match &self.inner {
            VInst::ConstZone(bits) => Ok(PyZoneAddr {
                inner: rs_addr::ZoneAddr::decode(*bits),
            }),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "zone_address() is only valid on const_zone",
            )),
        }
    }

    fn type_tag(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::NewArray(type_tag, ..) => Ok(*type_tag),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "type_tag() is only valid on new_array",
            )),
        }
    }

    fn dim0(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::NewArray(_, dim0, _) => Ok(*dim0),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "dim0() is only valid on new_array",
            )),
        }
    }

    fn dim1(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::NewArray(_, _, dim1) => Ok(*dim1),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "dim1() is only valid on new_array",
            )),
        }
    }

    fn ndims(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::GetItem(ndims) => Ok(*ndims),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "ndims() is only valid on get_item",
            )),
        }
    }

    fn arity(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::InitialFill(arity)
            | VInst::Fill(arity)
            | VInst::Move(arity)
            | VInst::LocalR(arity)
            | VInst::LocalRz(arity)
            | VInst::Measure(arity) => Ok(*arity),
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

/// Canonical mnemonic for a nested vihaco-cpu instruction. Only the ops the
/// lanes pipeline emits get a stable name; the rest fall back to `"cpu"`.
fn cpu_op_name(cpu: &Cpu) -> &'static str {
    match cpu {
        Cpu::Const(Value::F64(_)) => "const_float",
        Cpu::Const(Value::I64(_)) => "const_int",
        Cpu::Const(_) => "const",
        Cpu::Dup => "dup",
        Cpu::Halt => "halt",
        _ => "cpu",
    }
}

fn format_instruction(instr: &VInst) -> String {
    match instr {
        VInst::Pop => "Instruction.pop()".to_string(),
        VInst::Swap => "Instruction.swap()".to_string(),
        VInst::Return => "Instruction.return_()".to_string(),
        VInst::ConstLoc(bits) => {
            let addr = rs_addr::LocationAddr::decode(*bits);
            format!(
                "Instruction.const_loc(zone_id={}, word_id={}, site_id={})",
                addr.zone_id, addr.word_id, addr.site_id
            )
        }
        VInst::ConstLane(bits) => {
            let addr = rs_addr::LaneAddr::decode_u64(*bits);
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
        VInst::ConstZone(bits) => {
            let addr = rs_addr::ZoneAddr::decode(*bits);
            format!("Instruction.const_zone(zone_id={})", addr.zone_id)
        }
        VInst::InitialFill(arity) => format!("Instruction.initial_fill({arity})"),
        VInst::Fill(arity) => format!("Instruction.fill({arity})"),
        VInst::Move(arity) => format!("Instruction.move_({arity})"),
        VInst::LocalR(arity) => format!("Instruction.local_r({arity})"),
        VInst::LocalRz(arity) => format!("Instruction.local_rz({arity})"),
        VInst::GlobalR => "Instruction.global_r()".to_string(),
        VInst::GlobalRz => "Instruction.global_rz()".to_string(),
        VInst::Cz => "Instruction.cz()".to_string(),
        VInst::Measure(arity) => format!("Instruction.measure({arity})"),
        VInst::AwaitMeasure => "Instruction.await_measure()".to_string(),
        VInst::NewArray(type_tag, dim0, dim1) => {
            if *dim1 == 0 {
                format!("Instruction.new_array({type_tag}, {dim0})")
            } else {
                format!("Instruction.new_array({type_tag}, {dim0}, {dim1})")
            }
        }
        VInst::GetItem(ndims) => format!("Instruction.get_item({ndims})"),
        VInst::SetDetector => "Instruction.set_detector()".to_string(),
        VInst::SetObservable => "Instruction.set_observable()".to_string(),
        VInst::Cpu(Cpu::Const(Value::F64(f))) => format!("Instruction.const_float({f})"),
        VInst::Cpu(Cpu::Const(Value::I64(n))) => format!("Instruction.const_int({n})"),
        VInst::Cpu(Cpu::Dup) => "Instruction.dup()".to_string(),
        VInst::Cpu(Cpu::Halt) => "Instruction.halt()".to_string(),
        VInst::Cpu(other) => format!("Instruction.cpu({other:?})"),
    }
}
