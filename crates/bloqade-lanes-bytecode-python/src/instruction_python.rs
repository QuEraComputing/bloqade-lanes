use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr as rs_addr;
use bloqade_lanes_bytecode_core::isa::bytecode;
use bloqade_lanes_bytecode_core::isa::device::LanesInstruction as L;
use bloqade_lanes_bytecode_core::isa::machine::{self, MachineInstruction as VInst};
use vihaco::{Type, Value};
use vihaco_cpu::RuntimeInstruction as C;

use crate::arch_python::{PyDirection, PyLaneAddr, PyLocationAddr, PyMoveType, PyZoneAddr};
use crate::validation::validate_field;

#[pyclass(
    skip_from_py_object,
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

    #[staticmethod]
    fn const_float(value: f64) -> Self {
        Self {
            inner: VInst::Cpu(C::Const(Type::F64, Value::F64(value))),
        }
    }

    #[staticmethod]
    fn const_int(value: i64) -> Self {
        Self {
            inner: VInst::Cpu(C::Const(Type::I64, Value::I64(value))),
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
            inner: VInst::Lanes(L::ConstLoc(addr.encode())),
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
            inner: VInst::Lanes(L::ConstLane(addr.encode_u64())),
        })
    }

    #[staticmethod]
    fn const_zone(zone_id: i64) -> PyResult<Self> {
        let zone_id = validate_field::<u8>("zone_id", zone_id)? as u32;
        let addr = rs_addr::ZoneAddr { zone_id };
        Ok(Self {
            inner: VInst::Lanes(L::ConstZone(addr.encode())),
        })
    }

    // ── Stack manipulation ──

    #[staticmethod]
    fn dup() -> Self {
        Self {
            inner: VInst::Cpu(C::Dup),
        }
    }

    // ── Locals ──
    // There is no `pop` or `swap`: a function's locals are slots of their own
    // below its operands, so `store` parks a value out of their way and `load`
    // brings a copy back.

    /// Push a copy of local `index`, which holds a `value_type`.
    #[staticmethod]
    fn load(value_type: &str, index: i64) -> PyResult<Self> {
        let ty = parse_value_type(value_type)?;
        let index = validate_field::<u32>("index", index)?;
        Ok(Self {
            inner: VInst::Cpu(C::Load(ty, index)),
        })
    }

    /// Pop the top of the stack, a `value_type`, into local `index`.
    #[staticmethod]
    fn store(value_type: &str, index: i64) -> PyResult<Self> {
        let ty = parse_value_type(value_type)?;
        let index = validate_field::<u32>("index", index)?;
        Ok(Self {
            inner: VInst::Cpu(C::Store(ty, index)),
        })
    }

    // ── Atom operations ──

    #[staticmethod]
    fn initial_fill(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Lanes(L::InitialFill(arity)),
        })
    }

    #[staticmethod]
    fn fill(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Lanes(L::Fill(arity)),
        })
    }

    #[staticmethod]
    #[pyo3(name = "move_")]
    fn move_instr(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Lanes(L::Move(arity)),
        })
    }

    // ── Gate operations ──

    #[staticmethod]
    fn local_r(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Lanes(L::LocalR(arity)),
        })
    }

    #[staticmethod]
    fn local_rz(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Lanes(L::LocalRz(arity)),
        })
    }

    #[staticmethod]
    fn global_r() -> Self {
        Self {
            inner: VInst::Lanes(L::GlobalR),
        }
    }

    #[staticmethod]
    fn global_rz() -> Self {
        Self {
            inner: VInst::Lanes(L::GlobalRz),
        }
    }

    #[staticmethod]
    fn cz() -> Self {
        Self {
            inner: VInst::Lanes(L::Cz),
        }
    }

    // ── Measurement ──

    #[staticmethod]
    fn measure(arity: i64) -> PyResult<Self> {
        let arity = validate_field::<u32>("arity", arity)?;
        Ok(Self {
            inner: VInst::Lanes(L::Measure(arity)),
        })
    }

    #[staticmethod]
    fn await_measure() -> Self {
        Self {
            inner: VInst::Lanes(L::AwaitMeasure),
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
            inner: VInst::Lanes(L::NewArray(type_tag, dim0, dim1)),
        })
    }

    #[staticmethod]
    fn get_item(ndims: i64) -> PyResult<Self> {
        let ndims = validate_field::<u16>("ndims", ndims)? as u32;
        Ok(Self {
            inner: VInst::Lanes(L::GetItem(ndims)),
        })
    }

    // ── Detector / Observable ──

    #[staticmethod]
    fn set_detector() -> Self {
        Self {
            inner: VInst::Lanes(L::SetDetector),
        }
    }

    #[staticmethod]
    fn set_observable() -> Self {
        Self {
            inner: VInst::Lanes(L::SetObservable),
        }
    }

    // ── Control ──
    // `return` is lanes-native; `halt` is reused from vihaco-cpu.

    #[staticmethod]
    #[pyo3(name = "return_")]
    fn return_instr() -> Self {
        Self {
            inner: VInst::Cpu(C::Return(0)),
        }
    }

    #[staticmethod]
    fn halt() -> Self {
        Self {
            inner: VInst::Cpu(C::Halt),
        }
    }

    // ── Introspection ──

    /// The vihaco opcode byte for this instruction.
    #[getter]
    fn opcode(&self) -> u16 {
        bytecode::packed_opcode(&self.inner)
    }

    fn op_name(&self) -> &'static str {
        machine::op_name(&self.inner)
    }

    /// The device this instruction belongs to: `"cpu"` or `"lanes"`.
    fn device(&self) -> &'static str {
        machine::device_of(&self.inner)
    }

    fn float_value(&self) -> PyResult<f64> {
        match &self.inner {
            VInst::Cpu(C::Const(Type::F64, Value::F64(f))) => Ok(*f),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "float_value() is only valid on const_float",
            )),
        }
    }

    fn int_value(&self) -> PyResult<i64> {
        match &self.inner {
            VInst::Cpu(C::Const(Type::I64, Value::I64(n))) => Ok(*n),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "int_value() is only valid on const_int",
            )),
        }
    }

    fn location_address(&self) -> PyResult<PyLocationAddr> {
        match &self.inner {
            VInst::Lanes(L::ConstLoc(bits)) => Ok(PyLocationAddr {
                inner: rs_addr::LocationAddr::decode(*bits),
            }),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "location_address() is only valid on const_loc",
            )),
        }
    }

    fn lane_address(&self) -> PyResult<PyLaneAddr> {
        match &self.inner {
            VInst::Lanes(L::ConstLane(bits)) => Ok(PyLaneAddr {
                inner: rs_addr::LaneAddr::decode_u64(*bits),
            }),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "lane_address() is only valid on const_lane",
            )),
        }
    }

    fn zone_address(&self) -> PyResult<PyZoneAddr> {
        match &self.inner {
            VInst::Lanes(L::ConstZone(bits)) => Ok(PyZoneAddr {
                inner: rs_addr::ZoneAddr::decode(*bits),
            }),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "zone_address() is only valid on const_zone",
            )),
        }
    }

    fn type_tag(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::Lanes(L::NewArray(type_tag, ..)) => Ok(*type_tag),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "type_tag() is only valid on new_array",
            )),
        }
    }

    fn dim0(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::Lanes(L::NewArray(_, dim0, _)) => Ok(*dim0),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "dim0() is only valid on new_array",
            )),
        }
    }

    fn dim1(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::Lanes(L::NewArray(_, _, dim1)) => Ok(*dim1),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "dim1() is only valid on new_array",
            )),
        }
    }

    fn ndims(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::Lanes(L::GetItem(ndims)) => Ok(*ndims),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "ndims() is only valid on get_item",
            )),
        }
    }

    fn local_index(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::Cpu(C::Load(_, index) | C::Store(_, index)) => Ok(*index),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "local_index() is only valid on load and store",
            )),
        }
    }

    /// The type a `load`/`store` names, spelled as in the text format.
    fn value_type(&self) -> PyResult<&'static str> {
        match &self.inner {
            VInst::Cpu(C::Load(ty, _) | C::Store(ty, _)) => Ok(machine::cpu_type_text(*ty)),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "value_type() is only valid on load and store",
            )),
        }
    }

    fn arity(&self) -> PyResult<u32> {
        match &self.inner {
            VInst::Lanes(L::InitialFill(arity))
            | VInst::Lanes(L::Fill(arity))
            | VInst::Lanes(L::Move(arity))
            | VInst::Lanes(L::LocalR(arity))
            | VInst::Lanes(L::LocalRz(arity))
            | VInst::Lanes(L::Measure(arity)) => Ok(*arity),
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

/// Every vihaco value type, in declaration order.
const VALUE_TYPES: [Type; 9] = [
    Type::Undefined,
    Type::String,
    Type::Bool,
    Type::I64,
    Type::U32,
    Type::U64,
    Type::F64,
    Type::FunctionRef,
    Type::HeapRef,
];

/// A value type from its text-format spelling (`"u64"`, `"undef"`, …), so
/// Python spells a `load` the way the `.sst` it renders to does.
fn parse_value_type(name: &str) -> PyResult<Type> {
    VALUE_TYPES
        .into_iter()
        .find(|&ty| machine::cpu_type_text(ty) == name)
        .ok_or_else(|| {
            let names: Vec<&str> = VALUE_TYPES
                .into_iter()
                .map(machine::cpu_type_text)
                .collect();
            pyo3::exceptions::PyValueError::new_err(format!(
                "unknown value type {name:?}; expected one of {}",
                names.join(", ")
            ))
        })
}

fn format_instruction(instr: &VInst) -> String {
    match instr {
        VInst::Cpu(C::Return(0)) => "Instruction.return_()".to_string(),
        VInst::Lanes(L::ConstLoc(bits)) => {
            let addr = rs_addr::LocationAddr::decode(*bits);
            format!(
                "Instruction.const_loc(zone_id={}, word_id={}, site_id={})",
                addr.zone_id, addr.word_id, addr.site_id
            )
        }
        VInst::Lanes(L::ConstLane(bits)) => {
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
        VInst::Lanes(L::ConstZone(bits)) => {
            let addr = rs_addr::ZoneAddr::decode(*bits);
            format!("Instruction.const_zone(zone_id={})", addr.zone_id)
        }
        VInst::Lanes(L::InitialFill(arity)) => format!("Instruction.initial_fill({arity})"),
        VInst::Lanes(L::Fill(arity)) => format!("Instruction.fill({arity})"),
        VInst::Lanes(L::Move(arity)) => format!("Instruction.move_({arity})"),
        VInst::Lanes(L::LocalR(arity)) => format!("Instruction.local_r({arity})"),
        VInst::Lanes(L::LocalRz(arity)) => format!("Instruction.local_rz({arity})"),
        VInst::Lanes(L::GlobalR) => "Instruction.global_r()".to_string(),
        VInst::Lanes(L::GlobalRz) => "Instruction.global_rz()".to_string(),
        VInst::Lanes(L::Cz) => "Instruction.cz()".to_string(),
        VInst::Lanes(L::Measure(arity)) => format!("Instruction.measure({arity})"),
        VInst::Lanes(L::AwaitMeasure) => "Instruction.await_measure()".to_string(),
        VInst::Lanes(L::NewArray(type_tag, dim0, dim1)) => {
            if *dim1 == 0 {
                format!("Instruction.new_array({type_tag}, {dim0})")
            } else {
                format!("Instruction.new_array({type_tag}, {dim0}, {dim1})")
            }
        }
        VInst::Lanes(L::GetItem(ndims)) => format!("Instruction.get_item({ndims})"),
        VInst::Lanes(L::SetDetector) => "Instruction.set_detector()".to_string(),
        VInst::Lanes(L::SetObservable) => "Instruction.set_observable()".to_string(),
        VInst::Cpu(C::Const(Type::F64, Value::F64(f))) => format!("Instruction.const_float({f})"),
        VInst::Cpu(C::Const(Type::I64, Value::I64(n))) => format!("Instruction.const_int({n})"),
        VInst::Cpu(C::Dup) => "Instruction.dup()".to_string(),
        VInst::Cpu(C::Load(ty, index)) => format!(
            "Instruction.load({:?}, {index})",
            machine::cpu_type_text(*ty)
        ),
        VInst::Cpu(C::Store(ty, index)) => format!(
            "Instruction.store({:?}, {index})",
            machine::cpu_type_text(*ty)
        ),
        VInst::Cpu(C::Halt) => "Instruction.halt()".to_string(),
        // A decoded program can contain any vihaco-cpu op, but only the handful
        // above have Python factories. Emit the `.sst` spelling in a clearly
        // non-evaluable marker rather than a fake constructor call that
        // `repr()` would imply could be evaluated. Must come last: it matches
        // every CPU instruction.
        inst @ VInst::Cpu(_) => format!("<{}>", machine::to_sst_text(inst)),
    }
}
