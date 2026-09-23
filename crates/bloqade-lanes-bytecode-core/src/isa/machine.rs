//! The Bloqade Lanes machine: vihaco-cpu's `CPU` plus our atom-movement device.
//!
//! This is the idiomatic vihaco composition, and the same shape PPVM uses: each
//! device owns its instruction set, and `#[composite]` generates the combined
//! program type, the text parser (an alternation over the devices' parsers) and
//! the section loaders.
//!
//! ```text
//! cpu::cpu.const u64, 0      <- vihaco-cpu's device
//! lanes::lanes.move 2        <- ours
//! ```
//!
//! The `<field>::` prefix is the device field name; the second half is that
//! device's own dialect head.
//!
//! ## Who does what
//!
//! Instruction operands live on the CPU stack, so a device never reads them
//! directly. `resolve_lanes` pops them and packs them into a [`LanesMessage`];
//! the device executes; the machine then applies any [`LanesEffect::Push`]
//! back onto the stack. That round trip is the reason address constants can
//! stay lanes instructions while still behaving like stack pushes.
//!
//! [`super::validate::simulate_stack`] is the static half of this: it models
//! the same pops *and the same pushes*, so a program it accepts runs without
//! underflowing. The ops the device does not interpret hold up their end by
//! pushing [`Value::Undefined`] placeholders — the depth the simulator
//! predicts, with a value nothing can mistake for a result.

use vihaco::frame::Frame;
use vihaco::machine::StackFrame;
use vihaco::traits::StackMemory;
use vihaco::{Effects, GeneratedComponent, ProgramImage, Type, Value, composite};
use vihaco_cpu::{CPU, SurfaceInstruction as CpuSurfaceInstruction};

use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::types::ArchSpec;

use super::container::LanesContext;
use super::device::{Lanes, LanesEffect, LanesInstruction, LanesMessage};
use super::program::LanesInfo;
use super::program::Program;
use super::validate::{MAX_LOCAL_INDEX, array_element_count};

/// The combined instruction set: one variant per device.
pub type MachineInstruction = lanes_machine::runtime::Instruction;

/// The combined surface syntax, parsed from `.sst`.
pub type MachineSurfaceInstruction = lanes_machine::syntax::Instruction;

/// `#[composite]` derives only `Debug` and `Clone` on its runtime enum, but both
/// device instruction sets *are* `PartialEq`, so comparing programs only needs
/// the two arms spelled out. Round-trip tests and the Python `__eq__` both rely
/// on it.
impl PartialEq for MachineInstruction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu(a), Self::Cpu(b)) => a == b,
            (Self::Lanes(a), Self::Lanes(b)) => a == b,
            _ => false,
        }
    }
}

#[composite]
#[derive(Default)]
pub struct LanesMachine {
    /// Required by `#[composite]`: the loaded program plus its constants and
    /// device info. Unread until the bytecode/text layers are wired onto
    /// `MachineInstruction`.
    #[allow(dead_code)]
    loader: ProgramImage<MachineInstruction, LanesContext, Value, Type, LanesInfo>,

    #[device(0x00)]
    cpu: CPU,

    #[device(0x01)]
    lanes: Lanes,
}

/// Why [`LanesMachine::run`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stopped {
    /// The program executed `halt`.
    Halted,
    /// `@main` returned.
    Returned,
    /// Execution ran off the end of the code without a terminator.
    ///
    /// [`super::validate::validate_structure`] rejects such a program, so this
    /// only happens when running one that was never validated.
    RanOff,
    /// The step budget was exhausted. A program with a backward branch can
    /// run forever, and a machine with no budget would hang rather than say
    /// so.
    OutOfSteps,
}

/// What running a program produced.
#[derive(Debug, Clone)]
pub struct Run {
    /// Every effect the lanes device reported, in execution order. This is the
    /// record of what the program asked the hardware to do, and the input an
    /// observer for <https://github.com/QuEraComputing/bloqade-lanes/issues/1022>
    /// would consume.
    pub effects: Vec<LanesEffect>,
    /// Why execution stopped.
    pub stopped: Stopped,
    /// How many instructions ran.
    pub steps: u64,
}

impl LanesMachine {
    /// A machine ready to run, with the entry frame already pushed.
    ///
    /// vihaco's locals are a window into the operand stack starting at the
    /// current frame's `base`, so `load`/`store` and `ret` all fail with "no
    /// current frame" until one exists. A `call` pushes its own; `@main` is
    /// entered without one, so the machine establishes it.
    pub fn new() -> Self {
        let mut machine = Self::default();
        machine.cpu.push_frame(Frame {
            base: 0,
            span: (0, 0, 0),
            function: None,
            ret_pc: 0,
        });
        machine
    }

    /// Point the machine at an architecture. `move` cannot resolve a lane into
    /// (src, dst) endpoints without one, so it fails until this is set.
    pub fn with_arch(mut self, arch: ArchSpec) -> Self {
        self.lanes.arch = Some(arch);
        self
    }

    /// Run a program: dispatch each instruction to the device that owns it,
    /// until it halts, returns, or runs out of code or budget.
    ///
    /// This is the piece that makes the composite a machine rather than two
    /// devices: [`step_lanes`](Self::step_lanes) handles one lanes
    /// instruction, `CPU::execute_instruction` one CPU instruction, and
    /// nothing until now walked a `MachineInstruction` stream across both.
    ///
    /// `max_steps` bounds execution. A program with a backward branch can run
    /// forever, and the ISA carries vihaco-cpu's control flow whether or not
    /// the lanes compiler emits it, so the budget is a parameter rather than
    /// an assumption.
    ///
    /// The entry point is entered with no arguments; one that declares
    /// parameters needs [`run_with_args`](Self::run_with_args).
    pub fn run(&mut self, program: &Program, max_steps: u64) -> eyre::Result<Run> {
        self.run_with_args(program, &[], max_steps)
    }

    /// [`run`](Self::run) an entry point that takes arguments.
    ///
    /// `args` are pushed before the first instruction, in declaration order,
    /// so they sit at the bottom of the entry frame as locals `0..args.len()`
    /// — where a `call` leaves a callee's. That is the frame `simulate_stack`
    /// assumes for every function, the entry included.
    ///
    /// They are checked against the entry point's declaration, count and
    /// type, the way `validate` checks a `call` against its callee's. The host
    /// is the entry point's only caller, and without the check a missing
    /// argument surfaces mid-run as an out-of-bounds `load`, far from its
    /// cause.
    pub fn run_with_args(
        &mut self,
        program: &Program,
        args: &[Value],
        max_steps: u64,
    ) -> eyre::Result<Run> {
        use vihaco_cpu::StepOutcome;

        let mut effects = Vec::new();
        let mut steps = 0u64;
        let mut pc = self.enter(program, args)?;

        let stopped = loop {
            let Some(inst) = program.code.get(pc) else {
                break Stopped::RanOff;
            };
            if steps == max_steps {
                break Stopped::OutOfSteps;
            }
            steps += 1;

            // `op_call` reads this to work out where to return to, so it has
            // to be set before the instruction runs, not after.
            self.cpu.set_current_pc(pc as u32);

            match inst {
                MachineInstruction::Lanes(inst) => {
                    collect(self.step_lanes(inst.clone())?, &mut effects);
                }
                MachineInstruction::Cpu(inst) => {
                    self.guard_cpu(inst)?;
                    match self.cpu.execute_instruction(inst.clone())? {
                        StepOutcome::Halt => break Stopped::Halted,
                        // `op_return` reports `Return` only when it pops the
                        // last frame; an inner return sets the resume address
                        // and reports `Continue`.
                        StepOutcome::Return => break Stopped::Returned,
                        StepOutcome::Continue | StepOutcome::Breakpoint => {}
                    }
                }
            }

            // A branch or call leaves its destination here; anything else
            // falls through to the next instruction.
            pc = match self.cpu.take_pending_pc() {
                Some(target) => target as usize,
                None => pc + 1,
            };
        };

        Ok(Run {
            effects,
            stopped,
            steps,
        })
    }

    /// Refuse the two CPU instructions whose operands reach an allocation or a
    /// subtraction before vihaco-cpu 0.4.1 bounds-checks them.
    ///
    /// Both belong upstream, and both are caught here for the same reason
    /// [`pop_capacity`](Self::pop_capacity) clamps its reservation: `run` has
    /// no validation gate — `validate` is its own subcommand, and a program you
    /// have not validated is still one you may want to execute — so the
    /// execution path has to be safe on the operands alone.
    ///
    /// - `store` ([#1032]): `op_store` calls `get_local_mut`, which `resize`s
    ///   the operand stack to `base + index + 1`. The bound is the validator's
    ///   [`MAX_LOCAL_INDEX`], so a program the validator accepts still runs.
    /// - `ret` ([#1033]): `op_return` computes `stack.len() - frame.base`
    ///   before comparing it. `op_call` establishes `base <= len`, but a callee
    ///   that pops more than it pushed breaks that and the subtraction
    ///   underflows — a panic (debug) or an out-of-range `drain` (release)
    ///   rather than the `Err` an embedder can handle.
    ///
    /// Both are one comparison against state this loop already reads, and both
    /// can go when the `=0.4.1` pin moves past an upstream fix.
    ///
    /// [#1032]: https://github.com/QuEraComputing/bloqade-lanes/issues/1032
    /// [#1033]: https://github.com/QuEraComputing/bloqade-lanes/issues/1033
    fn guard_cpu(&self, inst: &vihaco_cpu::RuntimeInstruction) -> eyre::Result<()> {
        use vihaco_cpu::RuntimeInstruction as C;
        match inst {
            C::Store(_, index) if *index > MAX_LOCAL_INDEX => Err(eyre::eyre!(
                "store names local {index}, past the maximum index of {MAX_LOCAL_INDEX}"
            )),
            // A missing frame is upstream's own "no frame to pop", so only the
            // case it gets wrong is intercepted.
            C::Return(_) => match self.cpu.get_frame() {
                Ok(frame) if self.cpu.stack().len() < frame.base => Err(eyre::eyre!(
                    "ret with the stack {} deep, below its frame base {}: \
                     the callee popped past its own frame",
                    self.cpu.stack().len(),
                    frame.base
                )),
                _ => Ok(()),
            },
            _ => Ok(()),
        }
    }

    /// Where execution begins: the `@main` *symbol*, not byte zero.
    ///
    /// A lanes program is an executable, so it enters the way a linked binary
    /// does — at a named entry the loader resolves, wherever it was laid out.
    /// Starting at address 0 instead ran whichever function came first in the
    /// source, so a module declaring `@helper` before `@main` executed the
    /// wrong one and reported success.
    ///
    /// Pushes the entry point's arguments, checked against its declared
    /// parameters, and returns the address to start at.
    fn enter(&mut self, program: &Program, args: &[Value]) -> eyre::Result<usize> {
        let index = program
            .main_function
            .ok_or_else(|| eyre::eyre!("program declares no entry point"))?;
        let function = program.functions.get(index as usize).ok_or_else(|| {
            eyre::eyre!("entry point names function {index}, which the table does not have")
        })?;

        let params = &function.signature.params;
        if args.len() != params.len() {
            eyre::bail!(
                "the entry point declares {} parameter(s), but {} argument(s) were supplied",
                params.len(),
                args.len()
            );
        }
        for (i, (arg, param)) in args.iter().zip(params).enumerate() {
            if arg.type_of() != param.ty {
                eyre::bail!(
                    "entry argument {i} is {:?}, but the entry point declares {:?}",
                    arg.type_of(),
                    param.ty
                );
            }
        }
        for arg in args {
            self.cpu.stack_push(*arg);
        }

        Ok(function.start_address as usize)
    }

    /// Pop the operands `inst` consumes and pack them into its message.
    ///
    /// Pops are in reverse push order throughout: the last value pushed is the
    /// first popped, so a group read back from the stack is reversed to restore
    /// program order.
    fn resolve_lanes(&mut self, inst: &LanesInstruction) -> eyre::Result<LanesMessage> {
        use LanesInstruction as I;
        Ok(match inst {
            // Constants carry their operand in the instruction word.
            I::ConstLoc(_) | I::ConstLane(_) | I::ConstZone(_) => LanesMessage::None,

            // The device cannot reach the stack, so the machine does the
            // popping for it and the pushes come back as effects.
            I::Pop => {
                self.cpu.stack_pop()?;
                LanesMessage::None
            }
            I::Swap => LanesMessage::Values(self.pop_values(2)?),

            I::InitialFill(n) | I::Fill(n) => LanesMessage::Locations(self.pop_locations(*n)?),
            I::Move(n) => LanesMessage::Lanes(self.pop_lanes(*n)?),

            // Angles sit above the locations: `local_r` pops axis then
            // rotation, then the location group beneath them.
            I::LocalR(n) => LanesMessage::LocalRotation {
                angles: self.pop_floats(2)?,
                locations: self.pop_locations(*n)?,
            },
            I::LocalRz(n) => LanesMessage::LocalRotation {
                angles: self.pop_floats(1)?,
                locations: self.pop_locations(*n)?,
            },
            I::GlobalR => LanesMessage::GlobalRotation {
                angles: self.pop_floats(2)?,
            },
            I::GlobalRz => LanesMessage::GlobalRotation {
                angles: self.pop_floats(1)?,
            },

            I::Cz => LanesMessage::Zones(self.pop_zones(1)?),
            I::Measure(n) => LanesMessage::Zones(self.pop_zones(*n)?),

            // `new_array` consumes dim0×dim1 elements (dim1 = 0 means 1-D);
            // `get_item` consumes the array reference and then its indices.
            // Both are reported rather than executed — see
            // [`LanesEffect::NotSimulated`] — so the operands travel with the
            // message rather than being dropped on the floor.
            //
            // Both counts are computed in `u64`: `dim0 * dim1` overflows `u32`
            // for operands a decoded program is free to carry, and `n + 1`
            // overflows for `get_item(u32::MAX)`.
            I::NewArray(_, dim0, dim1) => {
                LanesMessage::Values(self.pop_values(array_element_count(*dim0, *dim1))?)
            }
            I::GetItem(n) => LanesMessage::Values(self.pop_values(*n as u64 + 1)?),

            I::AwaitMeasure | I::SetDetector | I::SetObservable => {
                LanesMessage::Values(self.pop_values(1)?)
            }
        })
    }

    /// Pop `n` values, restoring program order (the last pushed is popped
    /// first).
    ///
    /// `n` comes straight out of an instruction word, so nothing is
    /// pre-allocated against it: an implausible count runs out of stack within
    /// a few pops and fails there. [`super::validate::validate_structure`]
    /// rejects such a program up front; this makes the machine safe on its own
    /// regardless.
    fn pop_values(&mut self, n: u64) -> eyre::Result<Vec<Value>> {
        let mut out = Vec::with_capacity(self.pop_capacity(n));
        for _ in 0..n {
            out.push(self.cpu.stack_pop()?);
        }
        out.reverse();
        Ok(out)
    }

    /// Capacity to reserve for a pop of `n` values.
    ///
    /// `n` comes straight out of an instruction word, so it cannot be used as
    /// a capacity: `initial_fill 4294967295` asks for 48 GiB of
    /// `LocationAddr`. Linux's allocator refuses and Rust aborts the process
    /// (SIGABRT, no unwinding, no test failure to catch); macOS commits
    /// lazily and hands it back, which is why this only ever showed up in CI.
    ///
    /// A pop can never take more than the stack holds, so the stack depth is
    /// both a safe bound and a sufficient one — every legitimate arity still
    /// gets its single up-front allocation.
    fn pop_capacity(&self, n: u64) -> usize {
        n.min(self.cpu.stack().len() as u64) as usize
    }

    fn pop_u64(&mut self) -> eyre::Result<u64> {
        match self.cpu.stack_pop()? {
            Value::U64(v) => Ok(v),
            v => Err(eyre::eyre!("expected a packed u64 address, got {v:?}")),
        }
    }

    fn pop_floats(&mut self, n: u32) -> eyre::Result<Vec<f64>> {
        let mut out = Vec::with_capacity(self.pop_capacity(n as u64));
        for _ in 0..n {
            match self.cpu.stack_pop()? {
                Value::F64(v) => out.push(v),
                v => return Err(eyre::eyre!("expected an angle (f64), got {v:?}")),
            }
        }
        out.reverse();
        Ok(out)
    }

    fn pop_locations(&mut self, n: u32) -> eyre::Result<Vec<LocationAddr>> {
        let mut out = Vec::with_capacity(self.pop_capacity(n as u64));
        for _ in 0..n {
            out.push(LocationAddr::decode(self.pop_u64()?));
        }
        out.reverse();
        Ok(out)
    }

    fn pop_lanes(&mut self, n: u32) -> eyre::Result<Vec<LaneAddr>> {
        let mut out = Vec::with_capacity(self.pop_capacity(n as u64));
        for _ in 0..n {
            out.push(LaneAddr::decode_u64(self.pop_u64()?));
        }
        out.reverse();
        Ok(out)
    }

    fn pop_zones(&mut self, n: u32) -> eyre::Result<Vec<ZoneAddr>> {
        let mut out = Vec::with_capacity(self.pop_capacity(n as u64));
        for _ in 0..n {
            match self.cpu.stack_pop()? {
                Value::U32(v) => out.push(ZoneAddr::decode(v)),
                v => return Err(eyre::eyre!("expected a zone address (u32), got {v:?}")),
            }
        }
        out.reverse();
        Ok(out)
    }

    /// Apply a device effect. Only [`LanesEffect::Push`] touches the machine;
    /// the rest are observations for the caller.
    fn apply(&mut self, effects: &Effects<LanesEffect>) -> eyre::Result<()> {
        let mut push = |effect: &LanesEffect| {
            if let LanesEffect::Push(value) = effect {
                self.cpu.stack_push(*value);
            }
        };
        match effects {
            Effects::None => {}
            Effects::One(effect) => push(effect),
            Effects::Many(effects) => effects.iter().for_each(push),
        }
        Ok(())
    }

    /// Run one lanes instruction end to end: pop its operands, execute, then
    /// apply whatever comes back.
    pub fn step_lanes(&mut self, inst: LanesInstruction) -> eyre::Result<Effects<LanesEffect>> {
        let msg = self.resolve_lanes(&inst)?;
        let effects = self.lanes.execute_generated(inst, msg)?;
        self.apply(&effects)?;
        Ok(effects)
    }

    /// The atom arrangement as it currently stands.
    pub fn atoms(&self) -> &crate::atom_state::AtomStateData {
        &self.lanes.atoms
    }
}

/// Flatten one instruction's effects onto the run's record.
fn collect(effects: Effects<LanesEffect>, into: &mut Vec<LanesEffect>) {
    match effects {
        Effects::None => {}
        Effects::One(effect) => into.push(effect),
        Effects::Many(effects) => into.extend(effects),
    }
}

// ── Text rendering and introspection ──────────────────────────────────────────

/// vihaco-cpu's own `Display` emits bare mnemonics (`halt`, `const.f64 1.5`)
/// that its *parser* does not accept, so rendering is written here against the
/// surface grammar instead. The round-trip tests pin the two together.
pub(super) fn cpu_type_text(ty: Type) -> &'static str {
    match ty {
        Type::Undefined => "undef",
        Type::String => "str",
        Type::Bool => "bool",
        Type::I64 => "i64",
        Type::U32 => "u32",
        Type::U64 => "u64",
        Type::F64 => "f64",
        Type::FunctionRef => "fn_ref",
        Type::HeapRef => "heap_ref",
    }
}

fn cpu_value_text(value: &Value) -> String {
    match value {
        // `{:?}` on f64 is round-trip exact; `{}` drops the `.0` on integral
        // floats, which would then re-parse as an integer.
        Value::F64(v) => format!("{v:?}"),
        Value::Bool(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
        Value::U32(v) => v.to_string(),
        Value::U64(v) => v.to_string(),
        Value::String(v) | Value::FunctionRef(v) | Value::HeapRef(v) => v.to_string(),
        Value::Undefined => "undef".to_string(),
    }
}

fn cpu_text(inst: &vihaco_cpu::RuntimeInstruction) -> String {
    use vihaco_cpu::RuntimeInstruction as C;
    let typed = |op: &str, ty: Type| format!("{op} {}", cpu_type_text(ty));
    match inst {
        C::Span(a, b, c) => format!("span {a} {b} {c}"),
        C::Label(name) => format!("label @{}", name.as_str()),
        C::FunctionStart => "func_start".into(),
        C::FunctionEnd => "func_end".into(),
        C::Breakpoint => "breakpoint".into(),
        C::Branch(t) => format!("br @{t}"),
        C::ConditionalBranch(t, f) => format!("cond_br @{t}, @{f}"),
        C::Return(n) => format!("ret {n}"),
        C::IndirectCall => "call_indirect".into(),
        C::Call(arity, target) => format!("call {arity}, {target}"),
        C::Halt => "halt".into(),
        C::Print => "print".into(),
        C::Load(ty, addr) => format!("load {}, {addr}", cpu_type_text(*ty)),
        C::Store(ty, addr) => format!("store {}, {addr}", cpu_type_text(*ty)),
        C::Dup => "dup".into(),
        C::HeapAlloc(n) => format!("heap_alloc {n}"),
        C::GetItem => "get_item".into(),
        C::HeapDealloc => "heap_dealloc".into(),
        C::Const(ty, v) => format!("const {}, {}", cpu_type_text(*ty), cpu_value_text(v)),
        C::Add(t) => typed("add", *t),
        C::Sub(t) => typed("sub", *t),
        C::Mul(t) => typed("mul", *t),
        C::Div(t) => typed("div", *t),
        C::Rem(t) => typed("rem", *t),
        C::Neg(t) => typed("neg", *t),
        C::Shl(t) => typed("shl", *t),
        C::Shr(t) => typed("shr", *t),
        C::Rol(t) => typed("rol", *t),
        C::Ror(t) => typed("ror", *t),
        C::BitAnd(t) => typed("bitand", *t),
        C::BitOr(t) => typed("bitor", *t),
        C::BitXor(t) => typed("bitxor", *t),
        C::Not => "not".into(),
        C::And => "and".into(),
        C::Or => "or".into(),
        C::Xor => "xor".into(),
        C::Eq(t) => typed("eq", *t),
        C::Ne(t) => typed("ne", *t),
        C::Lt(t) => typed("lt", *t),
        C::Gt(t) => typed("gt", *t),
        C::Le(t) => typed("le", *t),
        C::Ge(t) => typed("ge", *t),
    }
}

fn lanes_text(inst: &LanesInstruction) -> String {
    use LanesInstruction as L;
    match inst {
        L::Pop => "pop".into(),
        L::Swap => "swap".into(),
        // Hex, fixed width, so addresses line up by eye.
        L::ConstLoc(v) => format!("const_loc 0x{v:016x}"),
        L::ConstLane(v) => format!("const_lane 0x{v:016x}"),
        L::ConstZone(v) => format!("const_zone 0x{v:08x}"),
        L::InitialFill(a) => format!("initial_fill {a}"),
        L::Fill(a) => format!("fill {a}"),
        L::Move(a) => format!("move {a}"),
        L::LocalRz(a) => format!("local_rz {a}"),
        L::LocalR(a) => format!("local_r {a}"),
        L::GlobalRz => "global_rz".into(),
        L::GlobalR => "global_r".into(),
        L::Cz => "cz".into(),
        L::Measure(a) => format!("measure {a}"),
        L::AwaitMeasure => "await_measure".into(),
        L::NewArray(t, d0, d1) => format!("new_array {t} {d0} {d1}"),
        L::GetItem(n) => format!("get_item {n}"),
        L::SetDetector => "set_detector".into(),
        L::SetObservable => "set_observable".into(),
    }
}

/// Render an instruction in the `.sst` surface grammar, including both the
/// device prefix and the dialect head — e.g. `lanes::lanes.move 2`.
pub fn to_sst_text(inst: &MachineInstruction) -> String {
    match inst {
        MachineInstruction::Cpu(i) => format!("cpu::cpu.{}", cpu_text(i)),
        MachineInstruction::Lanes(i) => format!("lanes::lanes.{}", lanes_text(i)),
    }
}

/// The device this instruction belongs to: `"cpu"` or `"lanes"`.
pub fn device_of(inst: &MachineInstruction) -> &'static str {
    match inst {
        MachineInstruction::Cpu(_) => "cpu",
        MachineInstruction::Lanes(_) => "lanes",
    }
}

/// Canonical opcode name, without any prefix — the key the Python decoder
/// dispatches on (`_visit_{op_name}`).
///
/// Two names deliberately differ from the text mnemonic, because the decoder
/// depends on them:
///
/// - the constants stay `const_float` / `const_int` rather than vihaco-cpu's
///   single typed `const`, since the decoder pushes a different value type for
///   each and the mnemonic alone would not say which;
/// - `return` keeps its spelling rather than vihaco-cpu's `ret`.
///
/// This is a `match` rather than the first token of [`to_sst_text`] because it
/// runs once per instruction in the Python decoder's loop, and rendering to
/// recover a name means formatting the operand that is then thrown away —
/// `const_loc` cost a 20-character hex format of its `u64` plus two
/// allocations, for one of about sixty fixed strings.
/// `every_op_name_matches_its_mnemonic` keeps it in step with the renderer.
pub fn op_name(inst: &MachineInstruction) -> &'static str {
    match inst {
        MachineInstruction::Cpu(i) => cpu_op_name(i),
        MachineInstruction::Lanes(i) => lanes_op_name(i),
    }
}

fn cpu_op_name(inst: &vihaco_cpu::RuntimeInstruction) -> &'static str {
    use vihaco_cpu::RuntimeInstruction as C;
    match inst {
        C::Span(..) => "span",
        C::Label(_) => "label",
        C::FunctionStart => "func_start",
        C::FunctionEnd => "func_end",
        C::Breakpoint => "breakpoint",
        C::Branch(_) => "br",
        C::ConditionalBranch(..) => "cond_br",
        C::Return(_) => "return",
        C::IndirectCall => "call_indirect",
        C::Call(..) => "call",
        C::Halt => "halt",
        C::Print => "print",
        C::Load(..) => "load",
        C::Store(..) => "store",
        C::Dup => "dup",
        C::HeapAlloc(_) => "heap_alloc",
        C::GetItem => "get_item",
        C::HeapDealloc => "heap_dealloc",
        C::Const(ty, _) => match ty {
            Type::Undefined => "const_undef",
            Type::String => "const_str",
            Type::Bool => "const_bool",
            Type::I64 => "const_int",
            Type::U32 => "const_u32",
            Type::U64 => "const_u64",
            Type::F64 => "const_float",
            Type::FunctionRef => "const_fn_ref",
            Type::HeapRef => "const_heap_ref",
        },
        C::Add(_) => "add",
        C::Sub(_) => "sub",
        C::Mul(_) => "mul",
        C::Div(_) => "div",
        C::Rem(_) => "rem",
        C::Neg(_) => "neg",
        C::Shl(_) => "shl",
        C::Shr(_) => "shr",
        C::Rol(_) => "rol",
        C::Ror(_) => "ror",
        C::BitAnd(_) => "bitand",
        C::BitOr(_) => "bitor",
        C::BitXor(_) => "bitxor",
        C::Not => "not",
        C::And => "and",
        C::Or => "or",
        C::Xor => "xor",
        C::Eq(_) => "eq",
        C::Ne(_) => "ne",
        C::Lt(_) => "lt",
        C::Gt(_) => "gt",
        C::Le(_) => "le",
        C::Ge(_) => "ge",
    }
}

fn lanes_op_name(inst: &LanesInstruction) -> &'static str {
    use LanesInstruction as L;
    match inst {
        L::Pop => "pop",
        L::Swap => "swap",
        L::ConstLoc(_) => "const_loc",
        L::ConstLane(_) => "const_lane",
        L::ConstZone(_) => "const_zone",
        L::InitialFill(_) => "initial_fill",
        L::Fill(_) => "fill",
        L::Move(_) => "move",
        L::LocalRz(_) => "local_rz",
        L::LocalR(_) => "local_r",
        L::GlobalRz => "global_rz",
        L::GlobalR => "global_r",
        L::Cz => "cz",
        L::Measure(_) => "measure",
        L::AwaitMeasure => "await_measure",
        L::NewArray(..) => "new_array",
        L::GetItem(_) => "get_item",
        L::SetDetector => "set_detector",
        L::SetObservable => "set_observable",
    }
}

/// Lower a parsed instruction to its executable form.
///
/// `#[composite]` generates the surface and runtime enums independently and no
/// conversion between them, so — as with the device — this is written out.
pub fn lower(inst: MachineSurfaceInstruction) -> eyre::Result<MachineInstruction> {
    Ok(match inst {
        MachineSurfaceInstruction::Cpu(i) => MachineInstruction::Cpu(lower_cpu(i)?),
        MachineSurfaceInstruction::Lanes(i) => MachineInstruction::Lanes(super::device::lower(i)),
    })
}

/// Lower a parsed vihaco-cpu instruction to its runtime form.
///
/// The surface form is lexical — `SurfaceValue` is a raw token and branch
/// targets are identifiers — so this is where a constant becomes a typed
/// [`Value`].
///
/// What this accepts is the counterpart of what [`to_sst_text`] emits: a
/// decoded binary may hold any of vihaco's nine `const` types, so all nine
/// parse back. The two rejections both need something a per-instruction
/// lowering does not have — a symbol table, for symbolic control flow, and a
/// string table, for a quoted literal.
fn lower_cpu(inst: CpuSurfaceInstruction) -> eyre::Result<vihaco_cpu::RuntimeInstruction> {
    use CpuSurfaceInstruction as S;
    use vihaco_cpu::RuntimeInstruction as R;

    let ty = |t: vihaco_cpu::SurfaceType| -> Type {
        match t {
            vihaco_cpu::SurfaceType::Undefined => Type::Undefined,
            vihaco_cpu::SurfaceType::String => Type::String,
            vihaco_cpu::SurfaceType::Bool => Type::Bool,
            vihaco_cpu::SurfaceType::I64 => Type::I64,
            vihaco_cpu::SurfaceType::U32 => Type::U32,
            vihaco_cpu::SurfaceType::U64 => Type::U64,
            vihaco_cpu::SurfaceType::F64 => Type::F64,
            vihaco_cpu::SurfaceType::FunctionRef => Type::FunctionRef,
            vihaco_cpu::SurfaceType::HeapRef => Type::HeapRef,
        }
    };

    Ok(match inst {
        S::Const(t, v) => {
            let ty = ty(t);
            // A quoted literal would have to be interned to become a
            // `Value::String`, and interning needs the module's string table —
            // which a per-instruction lowering does not have. `const str` takes
            // the interner index instead, which is what the value holds and
            // what `cpu_value_text` renders.
            let text = match &v {
                vihaco_cpu::SurfaceValue::Bare(token) => token.0.clone(),
                vihaco_cpu::SurfaceValue::Quoted(_) => {
                    return Err(eyre::eyre!(
                        "a quoted string literal needs the module's string table; \
                         write the interner index instead"
                    ));
                }
            };
            // Every type the renderer can emit must parse back, or
            // `disassemble` produces text `assemble` rejects. The three
            // reference types and `String` are interner/heap indices, so they
            // read as plain integers.
            let value = match ty {
                Type::F64 => Value::F64(text.parse()?),
                Type::I64 => Value::I64(text.parse()?),
                Type::U64 => Value::U64(text.parse()?),
                Type::U32 => Value::U32(text.parse()?),
                Type::Bool => Value::Bool(text.parse()?),
                Type::String => Value::String(text.parse()?),
                Type::FunctionRef => Value::FunctionRef(text.parse()?),
                Type::HeapRef => Value::HeapRef(text.parse()?),
                Type::Undefined if text == "undef" => Value::Undefined,
                Type::Undefined => {
                    return Err(eyre::eyre!("const undef takes no value but got '{text}'"));
                }
            };
            R::Const(ty, value)
        }
        S::Span(a, b, c) => R::Span(a, b, c),
        S::FunctionStart => R::FunctionStart,
        S::FunctionEnd => R::FunctionEnd,
        S::Breakpoint => R::Breakpoint,
        S::Return(n) => R::Return(n),
        S::IndirectCall => R::IndirectCall,
        S::Halt => R::Halt,
        S::Print => R::Print,
        S::Load(t, addr) => R::Load(ty(t), addr),
        S::Store(t, addr) => R::Store(ty(t), addr),
        S::Dup => R::Dup,
        S::HeapAlloc(n) => R::HeapAlloc(n),
        S::GetItem => R::GetItem,
        S::HeapDealloc => R::HeapDealloc,
        S::Add(t) => R::Add(ty(t)),
        S::Sub(t) => R::Sub(ty(t)),
        S::Mul(t) => R::Mul(ty(t)),
        S::Div(t) => R::Div(ty(t)),
        S::Rem(t) => R::Rem(ty(t)),
        S::Neg(t) => R::Neg(ty(t)),
        S::Shl(t) => R::Shl(ty(t)),
        S::Shr(t) => R::Shr(ty(t)),
        S::Rol(t) => R::Rol(ty(t)),
        S::Ror(t) => R::Ror(ty(t)),
        S::BitAnd(t) => R::BitAnd(ty(t)),
        S::BitOr(t) => R::BitOr(ty(t)),
        S::BitXor(t) => R::BitXor(ty(t)),
        S::Not => R::Not,
        S::And => R::And,
        S::Or => R::Or,
        S::Xor => R::Xor,
        S::Eq(t) => R::Eq(ty(t)),
        S::Ne(t) => R::Ne(ty(t)),
        S::Lt(t) => R::Lt(ty(t)),
        S::Gt(t) => R::Gt(ty(t)),
        S::Le(t) => R::Le(ty(t)),
        S::Ge(t) => R::Ge(ty(t)),
        S::Label(_) | S::Branch(_) | S::ConditionalBranch(_, _) | S::Call(_, _) => {
            return Err(eyre::eyre!(
                "symbolic control flow needs a label table and is not supported \
                 in a lanes program"
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::addr::LocationAddr;
    use crate::isa::device::LanesInstruction as I;
    use chumsky::Parser as _;
    use vihaco_parser::Parse;

    const SIMPLE_ARCH_JSON: &str = include_str!("../../../../examples/arch/simple.json");

    fn machine() -> LanesMachine {
        LanesMachine::new()
            .with_arch(ArchSpec::from_json(SIMPLE_ARCH_JSON).expect("simple.json should parse"))
    }

    fn loc(zone_id: u32, word_id: u32, site_id: u32) -> u64 {
        LocationAddr {
            zone_id,
            word_id,
            site_id,
        }
        .encode()
    }

    #[test]
    fn const_push_then_initial_fill_places_atoms() {
        // The whole point of the composite: address constants land on the CPU's
        // stack, and the lanes device reads them back off it.
        let mut m = machine();
        m.step_lanes(I::ConstLoc(loc(0, 0, 0))).unwrap();
        m.step_lanes(I::ConstLoc(loc(0, 0, 1))).unwrap();

        let effects = m.step_lanes(I::InitialFill(2)).unwrap();
        assert!(matches!(effects, Effects::One(LanesEffect::Arrangement(_))));

        // Both sites are now occupied, in program order.
        assert_eq!(
            m.atoms().get_qubit(&LocationAddr::decode(loc(0, 0, 0))),
            Some(0)
        );
        assert_eq!(
            m.atoms().get_qubit(&LocationAddr::decode(loc(0, 0, 1))),
            Some(1)
        );
    }

    #[test]
    fn initial_fill_underflows_without_enough_constants() {
        let mut m = machine();
        m.step_lanes(I::ConstLoc(loc(0, 0, 0))).unwrap();
        assert!(m.step_lanes(I::InitialFill(2)).is_err());
    }

    #[test]
    fn a_location_where_a_zone_belongs_is_rejected() {
        // `cz` wants a zone (u32); a location constant pushes a u64, so the
        // type mismatch surfaces at resolve time rather than silently decoding.
        let mut m = machine();
        m.step_lanes(I::ConstLoc(loc(0, 0, 0))).unwrap();
        let err = m.step_lanes(I::Cz).unwrap_err().to_string();
        assert!(err.contains("zone address"), "got {err}");
    }

    #[test]
    fn quantum_ops_are_reported_not_simulated() {
        let mut m = machine();
        m.step_lanes(I::ConstZone(0)).unwrap();
        let effects = m.step_lanes(I::Cz).unwrap();
        assert!(matches!(
            effects,
            Effects::One(LanesEffect::NotSimulated {
                inst: I::Cz,
                msg: LanesMessage::Zones(_)
            })
        ));
    }

    /// The entry frame is what makes locals addressable at all: without it every
    /// `load`/`store` fails with "no current frame".
    ///
    /// It also pins *what* a local is. vihaco locals are `stack[base + index]`
    /// — a window into the operand stack, not separate memory — so under the
    /// entry frame (`base = 0`) local 0 is literally stack slot 0. A function
    /// with no parameters therefore has no private locals, which is why `store`
    /// cannot stand in for `pop`: here it pops the operand and writes it
    /// straight back into slot 0.
    #[test]
    fn the_entry_frame_makes_locals_addressable_but_they_alias_the_stack() {
        use vihaco::traits::StackMemory;
        use vihaco_cpu::RuntimeInstruction as C;

        // Without a frame, locals are unreachable.
        let mut bare = LanesMachine::default();
        bare.cpu.stack_push(Value::U64(7));
        assert!(
            bare.cpu
                .execute_instruction(C::Store(Type::U64, 0))
                .unwrap_err()
                .to_string()
                .contains("no current frame")
        );

        // With one, the same store succeeds — and lands back in slot 0.
        let mut m = LanesMachine::new();
        m.cpu.stack_push(Value::U64(7));
        m.cpu.execute_instruction(C::Store(Type::U64, 0)).unwrap();
        assert_eq!(
            m.cpu.stack(),
            &[Value::U64(7)],
            "local 0 aliases stack slot 0 under the entry frame"
        );
        m.cpu.execute_instruction(C::Load(Type::U64, 0)).unwrap();
        assert_eq!(m.cpu.stack(), &[Value::U64(7), Value::U64(7)]);
    }

    #[test]
    fn swap_exchanges_the_top_two_and_pop_discards() {
        let mut m = machine();
        m.step_lanes(I::ConstZone(1)).unwrap();
        m.step_lanes(I::ConstZone(2)).unwrap();
        m.step_lanes(I::Swap).unwrap();

        // After the swap, `cz` consumes what was the *lower* of the two.
        let effects = m.step_lanes(I::Cz).unwrap();
        match effects {
            Effects::One(LanesEffect::NotSimulated {
                msg: LanesMessage::Zones(zones),
                ..
            }) => assert_eq!(zones[0].zone_id, 1),
            other => panic!("expected a zone message, got {other:?}"),
        }

        // `pop` discards, so the remaining value is gone and `cz` underflows.
        m.step_lanes(I::Pop).unwrap();
        assert!(m.step_lanes(I::Cz).is_err());
    }

    #[test]
    fn the_wrong_operand_type_is_reported_not_coerced() {
        // A zone where an angle belongs, and an angle where a location belongs.
        let mut m = machine();
        m.step_lanes(I::ConstZone(0)).unwrap();
        assert!(
            m.step_lanes(I::GlobalRz)
                .unwrap_err()
                .to_string()
                .contains("angle")
        );

        let mut m = machine();
        m.cpu.stack_push(Value::F64(1.0));
        assert!(
            m.step_lanes(I::InitialFill(1))
                .unwrap_err()
                .to_string()
                .contains("u64 address")
        );
    }

    #[test]
    fn device_and_op_name_identify_both_halves() {
        use vihaco_cpu::RuntimeInstruction as C;
        let cases = [
            (MachineInstruction::Lanes(I::Move(1)), "lanes", "move"),
            (MachineInstruction::Cpu(C::Halt), "cpu", "halt"),
            (MachineInstruction::Cpu(C::Return(0)), "cpu", "return"),
            (
                MachineInstruction::Cpu(C::Const(Type::F64, Value::F64(1.0))),
                "cpu",
                "const_float",
            ),
            (
                MachineInstruction::Cpu(C::Const(Type::I64, Value::I64(1))),
                "cpu",
                "const_int",
            ),
            (
                MachineInstruction::Cpu(C::Const(Type::U64, Value::U64(1))),
                "cpu",
                "const_u64",
            ),
            (MachineInstruction::Cpu(C::Add(Type::I64)), "cpu", "add"),
        ];
        for (inst, device, name) in cases {
            assert_eq!(device_of(&inst), device, "device for {inst:?}");
            assert_eq!(op_name(&inst), name, "op_name for {inst:?}");
        }
    }

    /// Every CPU instruction we can render must re-parse to itself.
    ///
    /// This is the pairing that has no compiler-enforced link: vihaco-cpu owns
    /// the parser, we own the renderer, and its own `Display` emits text its
    /// parser rejects (`halt`, not `cpu.halt`) — so nothing but a test keeps
    /// the two in step across all 42 ops.
    ///
    /// The samples come from [`every_instruction`], the one exhaustive list,
    /// rather than a second hand-written one. The list this replaced named
    /// five of vihaco's nine types, which read as exhaustive but left the four
    /// the lowering rejected — `undef`, `str`, `fn_ref`, `heap_ref` — untested,
    /// and a decoded binary can carry any of them.
    #[test]
    fn every_renderable_cpu_op_round_trips_through_text() {
        use crate::isa::bytecode::tests_support::every_instruction;
        use vihaco_cpu::RuntimeInstruction as C;

        let mut checked = 0;
        for inst in every_instruction() {
            let MachineInstruction::Cpu(ref cpu) = inst else {
                continue;
            };
            let text = to_sst_text(&inst);

            // TODO(#1025): `br`/`cond_br`/`call` render symbolically
            // (`br @L4`) but `lower_cpu` refuses them, because resolving a
            // label to an address needs a symbol table the per-instruction
            // lowering does not have. So these three render and do not
            // round-trip — a real asymmetry, asserted rather than skipped so
            // it is visible here and fails loudly once #1025's module-level
            // resolver closes it.
            if matches!(cpu, C::Branch(_) | C::ConditionalBranch(..) | C::Call(..)) {
                let parsed = MachineSurfaceInstruction::parser()
                    .parse(text.as_str())
                    .into_result()
                    .unwrap_or_else(|e| panic!("rendered {text:?} does not parse: {e:?}"));
                let err = lower(parsed)
                    .expect_err("#1025 has landed: this should now round-trip")
                    .to_string();
                assert!(err.contains("symbolic control flow"), "{text:?}: {err}");
                continue;
            }

            let parsed = MachineSurfaceInstruction::parser()
                .parse(text.as_str())
                .into_result()
                .unwrap_or_else(|e| panic!("rendered {text:?} does not parse: {e:?}"));
            let back = lower(parsed).unwrap_or_else(|e| panic!("{text:?} will not lower: {e}"));
            assert_eq!(back, inst, "round-trip changed {text:?}");
            checked += 1;
        }
        // A filter that silently matched nothing would make this vacuous.
        assert!(checked > 200, "only {checked} CPU instructions checked");
    }

    /// `op_name` is a hand-written table, so nothing but this keeps it in step
    /// with the renderer once an instruction is added or renamed.
    ///
    /// The three documented divergences are listed by name rather than waved
    /// at, so adding a fourth has to be deliberate.
    #[test]
    fn every_op_name_matches_its_mnemonic() {
        use crate::isa::bytecode::tests_support::every_instruction;
        for inst in every_instruction() {
            let mnemonic = to_sst_text(&inst)
                .split([' ', ','])
                .next()
                .unwrap()
                .rsplit('.')
                .next()
                .unwrap()
                .to_owned();
            let name = op_name(&inst);
            let known_divergence = match (mnemonic.as_str(), name) {
                // vihaco-cpu spells it `ret`; the decoder dispatches on
                // `_visit_return` and predates the rename.
                ("ret", "return") => true,
                // `const` is one typed instruction, but the decoder pushes a
                // different value type per type, so the name carries it.
                ("const", n) => n.starts_with("const_"),
                _ => false,
            };
            assert!(
                mnemonic == name || known_divergence,
                "op_name {name:?} does not match mnemonic {mnemonic:?} for {inst:?}"
            );
        }
    }

    // ── run ───────────────────────────────────────────────────────────────

    /// A whole program runs across both devices.
    ///
    /// Until `run` existed nothing walked a `MachineInstruction` stream: the
    /// tests pushed CPU operands by hand and stepped the lanes device alone,
    /// so the Cpu/Lanes dispatch had no caller and the execution layer was
    /// exercised only by its own unit tests.
    #[test]
    fn a_program_runs_across_both_devices() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        let program = from_code(
            Version::new(1, 0),
            vec![
                MachineInstruction::Lanes(I::ConstLoc(loc(0, 0, 0))),
                MachineInstruction::Lanes(I::ConstLoc(loc(0, 0, 1))),
                MachineInstruction::Lanes(I::InitialFill(2)),
                // A CPU constant feeding a lanes gate: the whole point of the
                // composite, and only reachable by dispatching both devices.
                MachineInstruction::Cpu(C::Const(Type::F64, Value::F64(1.5))),
                MachineInstruction::Lanes(I::GlobalRz),
                MachineInstruction::Cpu(C::Halt),
            ],
        )
        .unwrap();

        let mut m = machine();
        let run = m.run(&program, 100).unwrap();
        assert_eq!(run.stopped, Stopped::Halted);
        // Six instructions plus the `func_start` the entry point lands on.
        assert_eq!(run.steps, 7);

        // The atoms actually moved, and the gate was reported not simulated.
        assert_eq!(
            m.atoms().get_qubit(&LocationAddr::decode(loc(0, 0, 1))),
            Some(1)
        );
        assert!(run.effects.iter().any(|e| matches!(
            e,
            LanesEffect::NotSimulated {
                inst: I::GlobalRz,
                msg: LanesMessage::GlobalRotation { angles }
            } if angles == &[1.5]
        )));
    }

    /// An absurd operand count fails fast instead of doing absurd work.
    ///
    /// `run` deliberately has no validation gate — `validate` is its own
    /// subcommand, and a program you have not validated is still one you may
    /// want to execute. That is only safe if the pop path is bounded by what
    /// the program actually pushed rather than by what its instruction word
    /// claims.
    ///
    /// It was not. `pop_locations`/`pop_lanes`/`pop_zones`/`pop_floats`
    /// reserved `n` up front, so `initial_fill 4294967295` asked for 48 GiB
    /// and aborted the process on Linux. macOS commits lazily and returns the
    /// reservation, so the first version of this test passed locally and
    /// SIGABRTed in CI — hence the capacity assertion below, which fails on
    /// either platform.
    #[test]
    fn an_absurd_operand_count_fails_without_doing_the_work() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        // dim0 * dim1 = u32::MAX^2, about 1.8e19 elements.
        let cases = [
            MachineInstruction::Lanes(I::NewArray(0, u32::MAX, u32::MAX)),
            MachineInstruction::Lanes(I::GetItem(u32::MAX)),
            MachineInstruction::Lanes(I::InitialFill(u32::MAX)),
            MachineInstruction::Lanes(I::Measure(u32::MAX)),
        ];
        for inst in cases {
            // Three values on the stack, so the pop loop has somewhere to
            // start and must still stop at the fourth.
            let program = from_code(
                Version::new(1, 0),
                vec![
                    MachineInstruction::Cpu(C::Const(Type::I64, Value::I64(1))),
                    MachineInstruction::Cpu(C::Const(Type::I64, Value::I64(2))),
                    MachineInstruction::Cpu(C::Const(Type::I64, Value::I64(3))),
                    inst.clone(),
                    MachineInstruction::Cpu(C::Halt),
                ],
            )
            .unwrap();
            let mut machine = LanesMachine::new();
            let err = machine
                .run(&program, 100)
                .expect_err("{inst:?} should not run")
                .to_string();
            assert!(
                err.contains("stack") || err.contains("expected"),
                "{inst:?}: {err}"
            );

            // The reservation itself has to be bounded, not just the loop.
            // A platform that commits lazily will run the unfixed code
            // happily, so assert the bound rather than relying on the
            // allocator to complain.
            assert!(
                machine.pop_capacity(u32::MAX as u64) <= 3,
                "{inst:?}: capacity is not clamped to the stack depth"
            );
        }
    }

    /// A `store` cannot grow the operand stack to reach its index.
    ///
    /// `op_store` resizes the stack to `base + index + 1` and *writes* every
    /// new slot, so `store u64, 4294967295` makes ~68 GB resident — from a
    /// 12-byte program, and with no allocation failure to notice on a platform
    /// that commits lazily. `run` has no validation gate, so the assertion is
    /// on the stack depth rather than on the allocator complaining. See #1032.
    #[test]
    fn a_store_cannot_grow_the_stack_to_reach_its_index() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        for index in [MAX_LOCAL_INDEX + 1, 2_000_000, u32::MAX] {
            let program = from_code(
                Version::new(1, 0),
                vec![
                    MachineInstruction::Cpu(C::Const(Type::U64, Value::U64(7))),
                    MachineInstruction::Cpu(C::Store(Type::U64, index)),
                    MachineInstruction::Cpu(C::Halt),
                ],
            )
            .unwrap();
            let mut m = LanesMachine::new();
            let err = match m.run(&program, 100) {
                Ok(run) => panic!("index={index}: the store should have been refused, got {run:?}"),
                Err(e) => e.to_string(),
            };
            assert!(err.contains("local"), "index={index}: {err}");

            // The diagnosis is only worth anything if the allocation did not
            // happen first: one `const` pushed one value, and the refused
            // `store` must leave it at that.
            assert!(
                m.cpu.stack().len() <= 1,
                "index={index}: the stack grew to {} entries",
                m.cpu.stack().len()
            );
        }

        // An index inside the bound still stores, and still reads back.
        let program = from_code(
            Version::new(1, 0),
            vec![
                MachineInstruction::Cpu(C::Const(Type::U64, Value::U64(7))),
                MachineInstruction::Cpu(C::Store(Type::U64, MAX_LOCAL_INDEX)),
                MachineInstruction::Cpu(C::Load(Type::U64, MAX_LOCAL_INDEX)),
                MachineInstruction::Cpu(C::Halt),
            ],
        )
        .unwrap();
        let mut m = LanesMachine::new();
        assert_eq!(m.run(&program, 100).unwrap().stopped, Stopped::Halted);
        assert_eq!(m.cpu.stack().last(), Some(&Value::U64(7)));
    }

    /// A callee that pops below its own frame base gets an error, not a panic.
    ///
    /// `op_call` establishes `base <= stack.len()`, and vihaco-cpu 0.4.1's
    /// `op_return` computes `stack.len() - frame.base` assuming it stays that
    /// way. A callee that pops more than it pushed breaks the assumption: debug
    /// builds trap on the subtraction, release builds reach an out-of-range
    /// `drain`. Either way the CLI dies with a backtrace instead of a
    /// diagnosis, so `run` refuses the `ret` first. See #1033.
    #[test]
    fn a_callee_popping_below_its_frame_base_is_an_error_not_a_panic() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        let program = from_code(
            Version::new(1, 0),
            vec![
                // @main: three values, then a zero-arity call — so the callee's
                // frame base sits at 3 with nothing of its own beneath it.
                MachineInstruction::Lanes(I::ConstZone(0)),
                MachineInstruction::Lanes(I::ConstZone(1)),
                MachineInstruction::Lanes(I::ConstZone(2)),
                // Address 6: `from_code` prepends `@main`'s `func_start`, so
                // the callee's first instruction sits one past where a bare
                // instruction list would put it.
                MachineInstruction::Cpu(C::Call(0, 6)),
                MachineInstruction::Cpu(C::Halt),
                // @drain: pops the caller's three values, then returns.
                MachineInstruction::Lanes(I::Pop),
                MachineInstruction::Lanes(I::Pop),
                MachineInstruction::Lanes(I::Pop),
                MachineInstruction::Cpu(C::Return(0)),
            ],
        )
        .unwrap();
        let err = LanesMachine::new()
            .run(&program, 100)
            .expect_err("the underflowing ret should be refused")
            .to_string();
        assert!(err.contains("frame base"), "got {err}");
    }

    /// The guard must not change what a well-formed `call`/`ret` pair does —
    /// the underflow check is the only case it intercepts.
    #[test]
    fn a_balanced_call_still_returns_to_its_caller() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        let program = from_code(
            Version::new(1, 0),
            vec![
                MachineInstruction::Cpu(C::Const(Type::I64, Value::I64(1))),
                // One argument, so the callee's base is below it.
                MachineInstruction::Cpu(C::Call(1, 4)),
                MachineInstruction::Lanes(I::Pop),
                MachineInstruction::Cpu(C::Halt),
                // @callee: hands its argument back.
                MachineInstruction::Cpu(C::Return(1)),
            ],
        )
        .unwrap();
        let run = LanesMachine::new().run(&program, 100).unwrap();
        assert_eq!(run.stopped, Stopped::Halted);
    }

    /// What a nonzero-arity `call` and a `ret <keep>` actually *do*.
    ///
    /// The test above pins only that a balanced pair does not crash. The three
    /// behaviours the calling convention turns on go unobserved by it, and each
    /// fails differently:
    ///
    /// - the caller's operands become the callee's locals (`call <arity>` sets
    ///   `base = stack.len() - arity`, and locals index up from there);
    /// - `ret <keep>` keeps the top `keep` values and drains the rest of the
    ///   frame, so callee scratch goes and the return value survives;
    /// - the caller resumes at the instruction after the `call`, not at the
    ///   start of its own function.
    ///
    /// Each observable below is a distinct constant, so a failure names which
    /// one broke rather than just reporting a different stack.
    #[test]
    fn a_call_passes_locals_and_a_ret_keeps_only_what_it_says() {
        use crate::isa::text::parse_text;

        // `@callee` is declared first to keep the entry-point lookup honest.
        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n\
                   .header(root).\n.text(root):\n\
                   fn @callee() {\n  \
                     cpu::cpu.const i64, 777\n  \
                     cpu::cpu.load i64, 0\n  \
                     cpu::cpu.ret 1\n\
                   }\n\n\
                   fn @main() {\n  \
                     cpu::cpu.const i64, 111\n  \
                     cpu::cpu.const i64, 222\n  \
                     cpu::cpu.call 1, callee\n  \
                     cpu::cpu.const i64, 444\n  \
                     cpu::cpu.halt\n\
                   }\n\
                   .text(root).\n.section(root).\n";
        let program = parse_text(src).expect("the module should parse");

        let mut machine = LanesMachine::new();
        let run = machine.run(&program, 100).expect("the program should run");
        assert_eq!(run.stopped, Stopped::Halted);

        let stack = machine.cpu.stack();
        assert!(
            stack.contains(&Value::I64(222)),
            "the argument should have reached the callee as local 0 and come \
             back as its return value; stack: {stack:?}"
        );
        assert!(
            !stack.contains(&Value::I64(777)),
            "callee scratch below the kept value should be drained by `ret 1`; \
             stack: {stack:?}"
        );
        assert!(
            stack.contains(&Value::I64(444)),
            "the caller should resume at the instruction after the `call`; \
             stack: {stack:?}"
        );
        assert_eq!(
            stack,
            &[Value::I64(111), Value::I64(222), Value::I64(444)],
            "the caller's own operand below the frame base should be untouched"
        );
    }

    /// Execution enters at `@main`, wherever it was laid out.
    ///
    /// A lanes program is an executable, so the entry point is a symbol, not
    /// an address. Starting at 0 ran whichever function came first: a module
    /// declaring `@helper` before `@main` executed the helper's `ret` and
    /// reported success without touching a single atom.
    #[test]
    fn execution_enters_at_main_not_at_address_zero() {
        use crate::isa::text::parse_text;

        let program = parse_text(
            "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n             .text(root):\nfn @helper() {\n  cpu::cpu.ret 0\n}\n             fn @main() {\n  lanes::lanes.const_loc 0x0000000000000000\n               lanes::lanes.initial_fill 1\n  cpu::cpu.halt\n}\n             .text(root).\n.section(root).\n",
        )
        .expect("the module should parse");

        // `@main` is the second function, so its code starts past address 0.
        assert_eq!(program.main_function, Some(1));
        assert!(program.functions[1].start_address > 0);

        let mut m = machine();
        let run = m.run(&program, 100).unwrap();
        assert_eq!(run.stopped, Stopped::Halted, "should reach @main's halt");
        assert_eq!(
            run.steps, 4,
            "should run @main's `func_start` and its three instructions"
        );
        assert_eq!(
            m.atoms().get_qubit(&LocationAddr::decode(loc(0, 0, 0))),
            Some(0),
            "@main's initial_fill should have run"
        );
    }

    /// Declaration order is not part of a module's meaning.
    ///
    /// The test above shows `@main` is *found* when it is not first. This is
    /// the stronger property the entry-point symbol buys: the same two
    /// functions in either order are the same program, so reordering a module
    /// cannot change what it validates as or what it does.
    ///
    /// Worth pinning separately because the rules that could break it are not
    /// all in the entry-point lookup. `initial_fill must be first` and the
    /// reachability walk are per function only because `func_start` resets
    /// them; had either stayed whole-program, a helper declared first would
    /// have made `@main`'s own `initial_fill` illegal.
    #[test]
    fn declaration_order_changes_nothing() {
        use crate::isa::text::parse_text;
        use crate::isa::validate::{simulate_stack, validate, validate_structure};

        const HELPER: &str = "fn @helper() {\n  lanes::lanes.const_zone 0x00000000\n  \
             lanes::lanes.measure 1\n  lanes::lanes.await_measure\n  \
             lanes::lanes.pop\n  cpu::cpu.ret 0\n}\n";
        const MAIN: &str = "fn @main() {\n  lanes::lanes.const_loc 0x0000000000000000\n  \
             lanes::lanes.initial_fill 1\n  cpu::cpu.call 0, helper\n  cpu::cpu.halt\n}\n";

        let module = |body: String| {
            parse_text(&format!(
                "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n\
                 .header(root).\n.text(root):\n{body}.text(root).\n.section(root).\n"
            ))
            .expect("the module should parse")
        };

        let main_first = module(format!("{MAIN}\n{HELPER}"));
        let helper_first = module(format!("{HELPER}\n{MAIN}"));

        // The entry point moves; nothing that depends on it does.
        assert_eq!(main_first.main_function, Some(0));
        assert_eq!(helper_first.main_function, Some(1));

        for (label, program) in [("main first", &main_first), ("helper first", &helper_first)] {
            assert!(
                validate_structure(program).is_empty()
                    && validate(program, None).is_empty()
                    && simulate_stack(program, None).is_empty(),
                "{label}: should validate clean",
            );
        }

        let run_of = |program| {
            let mut m = machine();
            let run = m.run(program, 100).expect("the program should run");
            (run.stopped, run.steps, run.effects.len())
        };
        assert_eq!(
            run_of(&main_first),
            run_of(&helper_first),
            "the same functions in either order should execute identically"
        );
    }

    /// An entry point that takes one `u32`, reads it, and measures it.
    fn parameterised_main() -> Program {
        crate::isa::text::parse_text(
            "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n\
             .text(root):\nfn @main(z: u32) {\n  cpu::cpu.load u32, 0\n  \
             lanes::lanes.measure 1\n  lanes::lanes.await_measure\n  lanes::lanes.pop\n  \
             lanes::lanes.pop\n  cpu::cpu.halt\n}\n.text(root).\n.section(root).\n",
        )
        .expect("the module should parse")
    }

    /// The host is the entry point's caller. Its arguments land at the bottom
    /// of the entry frame, where `load` finds them — and where
    /// `simulate_stack` seeds every function's declared parameters, the entry
    /// included.
    #[test]
    fn an_entry_point_receives_its_arguments() {
        use crate::isa::validate::{simulate_stack, validate_structure};

        let program = parameterised_main();
        assert_eq!(validate_structure(&program), vec![]);
        assert_eq!(simulate_stack(&program, None), vec![]);

        let run = machine()
            .run_with_args(&program, &[Value::U32(0)], 100)
            .expect("the program should run");
        assert_eq!(run.stopped, Stopped::Halted);
    }

    /// A missing argument is refused at entry, naming the cause, rather than
    /// surfacing mid-run as an out-of-bounds `load`.
    #[test]
    fn a_missing_entry_argument_is_refused_at_entry() {
        let error = machine()
            .run(&parameterised_main(), 100)
            .expect_err("@main declares a parameter nothing supplied");
        assert!(
            error
                .to_string()
                .contains("declares 1 parameter(s), but 0 argument(s) were supplied"),
            "got: {error}"
        );

        // And the other way: arguments for an entry point that takes none.
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;
        let program =
            from_code(Version::new(1, 0), vec![MachineInstruction::Cpu(C::Halt)]).unwrap();
        let error = machine()
            .run_with_args(&program, &[Value::U32(0)], 100)
            .expect_err("@main takes no arguments");
        assert!(
            error
                .to_string()
                .contains("declares 0 parameter(s), but 1 argument(s) were supplied"),
            "got: {error}"
        );
    }

    /// Arguments are checked against the declared types, as `load` would
    /// check them later — but here, at the boundary that supplied them.
    #[test]
    fn a_mistyped_entry_argument_is_refused_at_entry() {
        let error = machine()
            .run_with_args(&parameterised_main(), &[Value::I64(0)], 100)
            .expect_err("the argument is not the declared u32");
        assert!(
            error.to_string().contains("entry argument 0 is I64"),
            "got: {error}"
        );
    }

    /// `ret` at top level ends the program, which needs the entry frame:
    /// `op_return` pops a frame and errors when there is none.
    #[test]
    fn main_returning_stops_the_program() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        let program = from_code(
            Version::new(1, 0),
            vec![MachineInstruction::Cpu(C::Return(0))],
        )
        .unwrap();
        let run = LanesMachine::new().run(&program, 100).unwrap();
        assert_eq!(run.stopped, Stopped::Returned);
    }

    /// The ops the device does not simulate must still leave the stack at the
    /// depth `simulate_stack` predicts, or a *validated* program underflows
    /// the moment it runs.
    #[test]
    fn a_validated_measurement_pipeline_runs_without_underflowing() {
        use crate::isa::program::from_code;
        use crate::isa::validate::simulate_stack;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        let program = from_code(
            Version::new(1, 0),
            vec![
                MachineInstruction::Lanes(I::ConstZone(0)),
                MachineInstruction::Lanes(I::Measure(1)),
                MachineInstruction::Lanes(I::AwaitMeasure),
                MachineInstruction::Lanes(I::SetDetector),
                MachineInstruction::Lanes(I::Pop),
                MachineInstruction::Cpu(C::Halt),
            ],
        )
        .unwrap();
        // The static half accepts it...
        assert_eq!(simulate_stack(&program, None), vec![]);
        // ...so running it must not underflow.
        let run = machine().run(&program, 100).unwrap();
        assert_eq!(run.stopped, Stopped::Halted);
    }

    /// `set_detector` must report *which* array it referenced, not just that
    /// it happened — that is the whole point of the `NotSimulated` effect.
    #[test]
    fn a_not_simulated_op_reports_the_operands_it_consumed() {
        let mut m = machine();
        m.cpu.stack_push(Value::U32(7));
        let effects = m.step_lanes(I::SetDetector).unwrap();
        assert!(matches!(
            effects,
            Effects::Many(ref e) if matches!(
                &e[0],
                LanesEffect::NotSimulated { msg: LanesMessage::Values(v), .. }
                    if v == &[Value::U32(7)]
            )
        ));
    }

    /// A backward branch is a loop, and the budget is what keeps it from
    /// being a hang.
    #[test]
    fn a_looping_program_runs_out_of_steps() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco_cpu::RuntimeInstruction as C;

        let program = from_code(
            Version::new(1, 0),
            vec![MachineInstruction::Cpu(C::Branch(0))],
        )
        .unwrap();
        let run = LanesMachine::new().run(&program, 50).unwrap();
        assert_eq!(run.stopped, Stopped::OutOfSteps);
        assert_eq!(run.steps, 50);
    }

    /// Running off the end is reported rather than silently treated as a halt
    /// — `validate_structure` rejects such a program, so reaching it means
    /// something skipped validation.
    #[test]
    fn a_program_without_a_terminator_runs_off_the_end() {
        use crate::isa::program::from_code;
        use crate::version::Version;

        let program = from_code(
            Version::new(1, 0),
            vec![MachineInstruction::Lanes(I::ConstZone(0))],
        )
        .unwrap();
        let run = LanesMachine::new().run(&program, 100).unwrap();
        assert_eq!(run.stopped, Stopped::RanOff);
    }

    /// The docs that name instructions, read at compile time so a rename
    /// cannot leave them behind.
    const DOCS: [(&str, &str); 3] = [
        (
            "docs/src/bytecode/inst-quick-ref.md",
            include_str!("../../../../docs/src/bytecode/inst-quick-ref.md"),
        ),
        (
            "docs/src/bytecode/inst-spec.md",
            include_str!("../../../../docs/src/bytecode/inst-spec.md"),
        ),
        (
            "docs/src/migration/migration_guide_0_12.md",
            include_str!("../../../../docs/src/migration/migration_guide_0_12.md"),
        ),
    ];

    /// Every text spelling the machine has, mapped to its packed opcode.
    fn real_mnemonics() -> std::collections::BTreeMap<String, u16> {
        use crate::isa::bytecode::{packed_opcode, tests_support::every_instruction};
        use vihaco_cpu::RuntimeInstruction as C;

        // `label` is excluded from the exhaustive list because it has no
        // encodable form, but it is still a spelling the docs may use.
        every_instruction()
            .iter()
            .chain(std::iter::once(&MachineInstruction::Cpu(C::Label(
                vihaco_parser::Ident("l".into()),
            ))))
            .map(|i| {
                (
                    to_sst_text(i).split([' ', ',']).next().unwrap().to_owned(),
                    packed_opcode(i),
                )
            })
            .collect()
    }

    /// A full text spelling is `<device>::<dialect>.<mnemonic>`. The narrower
    /// `cpu::add` form also appears in the docs — it is how a `DecodingError`
    /// names an instruction — but that is `op_name`, not a text mnemonic.
    fn is_text_spelling(token: &str) -> bool {
        token
            .split_once("::")
            .and_then(|(device, rest)| {
                let (dialect, mnemonic) = rest.split_once('.')?;
                Some(
                    matches!(device, "cpu" | "lanes")
                        && matches!(dialect, "cpu" | "lanes")
                        && !mnemonic.is_empty(),
                )
            })
            .unwrap_or(false)
    }

    /// The first text spelling in `line`, if any.
    fn mnemonic_in(line: &str) -> Option<&str> {
        line.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':' || c == '.'))
            .map(|t| t.trim_end_matches('.'))
            .find(|t| is_text_spelling(t))
    }

    /// The first `0x`-prefixed hex literal in `line`, if any.
    fn hex_in(line: &str) -> Option<u16> {
        line.split(|c: char| !(c.is_alphanumeric() || c == 'x'))
            .find_map(|t| t.strip_prefix("0x"))
            .and_then(|t| u16::from_str_radix(t, 16).ok())
    }

    /// Every `<device>::<dialect>.<mnemonic>` the docs print must be real.
    ///
    /// `inst-spec.md` documented `cpu::cpu.const_int`, `cpu::cpu.const_float`
    /// and `cpu::cpu.return`, none of which parse: the constants are one typed
    /// `const` taking a comma, and `return` is spelled `ret`. Prose drifts
    /// silently, so the docs are read here rather than trusted.
    #[test]
    fn every_documented_mnemonic_exists() {
        let mut bogus = Vec::new();
        for (file, text) in DOCS {
            let tokens =
                text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':' || c == '.'));
            for token in tokens {
                let token = token.trim_end_matches('.');
                if is_text_spelling(token) && !real_mnemonics().contains_key(token) {
                    bogus.push(format!("  {file}: `{token}`"));
                }
            }
        }
        bogus.sort();
        bogus.dedup();
        assert!(
            bogus.is_empty(),
            "the docs name {} mnemonic(s) the machine does not have:\n{}",
            bogus.len(),
            bogus.join("\n")
        );
    }

    /// Every `Opcode` the docs tabulate must be the one the encoder produces.
    ///
    /// The spec prints a literal for each instruction, and nothing pinned
    /// them: they are assigned by declaration order, so they renumber whenever
    /// either instruction set gains a variant. The guide tells readers to
    /// compare on `op_name()` for exactly that reason — but a *format spec*
    /// that omits the byte values is not much of a format spec, so they stay
    /// and are checked here instead.
    #[test]
    fn every_documented_opcode_matches_the_encoding() {
        let real = real_mnemonics();
        let mut wrong = Vec::new();
        let mut checked = 0;

        for (file, text) in DOCS {
            // `inst-spec.md` puts the mnemonic in a `#### ` heading and the
            // value in an `| Opcode | 0x.. |` row; `inst-quick-ref.md` puts
            // both in one row. Track the heading so either shape resolves.
            let mut heading: Option<&str> = None;
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("#### ") {
                    heading = mnemonic_in(rest);
                    continue;
                }
                if !line.starts_with('|') {
                    continue;
                }
                let Some(documented) = hex_in(line) else {
                    continue;
                };
                // Either the row names the instruction itself, or it is the
                // `Opcode` row belonging to the heading above it. Anything
                // else carrying a hex literal (the device-code table) is not
                // an opcode claim.
                let first_cell = line.split('|').nth(1).map(str::trim).unwrap_or("");
                let Some(name) = mnemonic_in(line).or(if first_cell == "Opcode" {
                    heading
                } else {
                    None
                }) else {
                    continue;
                };

                match real.get(name) {
                    Some(actual) if *actual == documented => checked += 1,
                    Some(actual) => wrong.push(format!(
                        "  {file}: `{name}` documented as {documented:#06x}, encodes as {actual:#06x}"
                    )),
                    // The mnemonic test covers unknown names.
                    None => {}
                }
            }
        }

        assert!(
            wrong.is_empty(),
            "{} documented opcode(s) disagree with the encoding:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
        // A parser that silently matched nothing would make this vacuous:
        // 29 rows in the quick reference plus 23 in the spec.
        assert_eq!(
            checked, 52,
            "expected 52 documented opcodes, found {checked}"
        );
    }

    /// Symbolic control flow renders, but cannot be lowered without a label
    /// table — the error says so rather than silently inventing an address.
    #[test]
    fn symbolic_control_flow_is_rejected_on_lowering() {
        use vihaco_cpu::SurfaceInstruction as S;
        for inst in [
            S::Branch(vihaco_parser::Ident("loop".into())),
            S::Label(vihaco_parser::Ident("loop".into())),
        ] {
            let err = lower(MachineSurfaceInstruction::Cpu(inst))
                .unwrap_err()
                .to_string();
            assert!(err.contains("label table"), "got {err}");
        }
    }

    /// Every lanes instruction must likewise survive render -> parse, and
    /// from the same exhaustive list as the CPU half.
    #[test]
    fn every_lanes_op_round_trips_through_text() {
        use crate::isa::bytecode::tests_support::every_instruction;

        let mut checked = 0;
        for inst in every_instruction() {
            if !matches!(inst, MachineInstruction::Lanes(_)) {
                continue;
            }
            let text = to_sst_text(&inst);
            let parsed = MachineSurfaceInstruction::parser()
                .parse(text.as_str())
                .into_result()
                .unwrap_or_else(|e| panic!("rendered {text:?} does not parse: {e:?}"));
            assert_eq!(lower(parsed).unwrap(), inst, "round-trip changed {text:?}");
            checked += 1;
        }
        assert_eq!(
            checked, 19,
            "the lanes device has 19 instructions; every_instruction() yielded {checked}"
        );
    }

    #[test]
    fn move_without_an_arch_is_an_error() {
        // `move` cannot resolve a lane into endpoints without a spec.
        let mut m = LanesMachine::new();
        m.step_lanes(I::ConstLane(0)).unwrap();
        let err = m.step_lanes(I::Move(1)).unwrap_err().to_string();
        assert!(err.contains("arch spec"), "got {err}");
    }
}
