//! Lower parsed functions into a [`Program`]: symbols, labels, addresses.
//!
//! Parsing gives back functions whose control flow is still *symbolic* — `br
//! @loop`, `call 2, @helper`. Turning those into addresses needs the whole
//! module in view, which is what this module does and why per-instruction
//! lowering ([`machine::lower`]) rejects them.
//!
//! ## Two passes
//!
//! 1. **Lay out.** Walk every function in order, emitting its body into one
//!    flat code vector and recording where each function and label landed.
//!    Control-flow operands are emitted as `0` and a fixup is recorded, because
//!    a forward branch names something not yet placed.
//! 2. **Patch.** Resolve each fixup against the label and function tables.
//!
//! ## Labels are metadata, not instructions
//!
//! vihaco executes `Label` as a no-op (`Label(_) => Continue`) — it exists only
//! to mark a position. It also carries an `Ident`, which has no meaning outside
//! the parse that produced it and therefore no encodable form.
//!
//! So labels do not survive into the code stream: the resolver records each one
//! in [`LabelInfo`] against the address of the instruction that follows it, and
//! drops it. Addresses are computed after the drop, so they stay consistent, and
//! every instruction in `code` is encodable. [`super::text::to_text`] re-emits
//! the labels from the table.
//!
//! One divergence worth naming: vihaco's `FunctionInfo::start_address` is
//! documented as "corresponds to a label noop". Ours is simply the index of the
//! function's first instruction, since we keep no label noops. Nothing in vihaco
//! depends on this today — we drive execution ourselves rather than through its
//! loader — but it would matter if that changed.

use vihaco::module::{FunctionInfo, LabelInfo, Signature};
use vihaco::syntax::ParsedFunction;
use vihaco_cpu::SurfaceInstruction as CpuSurface;

use super::machine::{self, MachineInstruction, MachineSurfaceInstruction};
use super::program::{LanesInfo, Program};
use super::text::NoType;

/// A failure to resolve a parsed module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Two labels share a name. Labels are module-global (vihaco's `LabelInfo`
    /// carries no function), so a repeat is ambiguous rather than shadowing.
    DuplicateLabel { name: String },
    /// Two functions share a name.
    DuplicateFunction { name: String },
    /// `br`/`cond_br` names a label that does not exist.
    UnknownLabel { name: String },
    /// `call` names a function that does not exist.
    UnknownFunction { name: String },
    /// The module declares no `@main`.
    MissingMain,
    /// An instruction could not be lowered (see [`machine::lower`]).
    Lowering { message: String },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::DuplicateLabel { name } => write!(f, "duplicate label @{name}"),
            ResolveError::DuplicateFunction { name } => write!(f, "duplicate function @{name}"),
            ResolveError::UnknownLabel { name } => write!(f, "branch to unknown label @{name}"),
            ResolveError::UnknownFunction { name } => write!(f, "call to unknown function @{name}"),
            ResolveError::MissingMain => write!(f, "module declares no @main"),
            ResolveError::Lowering { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// A control-flow operand that could not be filled in during layout.
#[derive(Debug)]
enum Fixup {
    Branch {
        at: usize,
        target: String,
    },
    ConditionalBranch {
        at: usize,
        on_true: String,
        on_false: String,
    },
    Call {
        at: usize,
        callee: String,
    },
}

/// Interns strings, so `FunctionInfo`/`LabelInfo` can hold indices as vihaco
/// expects rather than owned names.
#[derive(Default)]
struct Interner {
    strings: Vec<String>,
}

impl Interner {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(i) = self.strings.iter().position(|existing| existing == s) {
            return i as u32;
        }
        self.strings.push(s.to_owned());
        (self.strings.len() - 1) as u32
    }
}

/// Lower parsed functions into a [`Program`].
pub fn resolve(
    functions: Vec<ParsedFunction<MachineSurfaceInstruction, NoType>>,
    extra: LanesInfo,
) -> Result<Program, ResolveError> {
    let mut interner = Interner::default();
    let mut code: Vec<MachineInstruction> = Vec::new();
    let mut labels: Vec<LabelInfo> = Vec::new();
    let mut label_addresses: Vec<(String, u32)> = Vec::new();
    let mut function_addresses: Vec<(String, u32)> = Vec::new();
    let mut function_infos: Vec<FunctionInfo<vihaco::Type>> = Vec::new();
    let mut fixups: Vec<Fixup> = Vec::new();
    let mut main_function: Option<u32> = None;

    // ── Pass 1: lay out ──
    for func in &functions {
        let name = func.name.as_str().to_owned();
        if function_addresses.iter().any(|(n, _)| *n == name) {
            return Err(ResolveError::DuplicateFunction { name });
        }
        let start_address = code.len() as u32;
        function_addresses.push((name.clone(), start_address));
        if name == "main" {
            main_function = Some(function_infos.len() as u32);
        }

        for inst in &func.body {
            match inst {
                // A label marks the address of whatever comes next, and is not
                // itself emitted.
                MachineSurfaceInstruction::Cpu(CpuSurface::Label(ident)) => {
                    let label = ident.as_str().to_owned();
                    if label_addresses.iter().any(|(n, _)| *n == label) {
                        return Err(ResolveError::DuplicateLabel { name: label });
                    }
                    let address = code.len() as u32;
                    label_addresses.push((label.clone(), address));
                    labels.push(LabelInfo {
                        address,
                        name: interner.intern(&label),
                    });
                }

                // Symbolic control flow: emit a placeholder and note the fixup.
                MachineSurfaceInstruction::Cpu(CpuSurface::Branch(target)) => {
                    fixups.push(Fixup::Branch {
                        at: code.len(),
                        target: target.as_str().to_owned(),
                    });
                    code.push(MachineInstruction::Cpu(
                        vihaco_cpu::RuntimeInstruction::Branch(0),
                    ));
                }
                MachineSurfaceInstruction::Cpu(CpuSurface::ConditionalBranch(t, f)) => {
                    fixups.push(Fixup::ConditionalBranch {
                        at: code.len(),
                        on_true: t.as_str().to_owned(),
                        on_false: f.as_str().to_owned(),
                    });
                    code.push(MachineInstruction::Cpu(
                        vihaco_cpu::RuntimeInstruction::ConditionalBranch(0, 0),
                    ));
                }
                MachineSurfaceInstruction::Cpu(CpuSurface::Call(arity, callee)) => {
                    fixups.push(Fixup::Call {
                        at: code.len(),
                        callee: callee.as_str().to_owned(),
                    });
                    code.push(MachineInstruction::Cpu(
                        vihaco_cpu::RuntimeInstruction::Call(*arity, 0),
                    ));
                }

                // Everything else lowers on its own.
                other => code.push(machine::lower(other.clone()).map_err(|e| {
                    ResolveError::Lowering {
                        message: e.to_string(),
                    }
                })?),
            }
        }

        function_infos.push(FunctionInfo {
            name: interner.intern(&name),
            signature: Signature {
                params: Vec::new(),
                ret: Vec::new(),
            },
            local_count: 0,
            start_address,
            end_address: code.len() as u32,
            file: 0,
        });
    }

    // ── Pass 2: patch ──
    let label_of = |name: &str| -> Result<u32, ResolveError> {
        label_addresses
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| *a)
            .ok_or_else(|| ResolveError::UnknownLabel {
                name: name.to_owned(),
            })
    };
    let function_of = |name: &str| -> Result<u32, ResolveError> {
        function_addresses
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| *a)
            .ok_or_else(|| ResolveError::UnknownFunction {
                name: name.to_owned(),
            })
    };

    for fixup in fixups {
        match fixup {
            Fixup::Branch { at, target } => {
                code[at] = MachineInstruction::Cpu(vihaco_cpu::RuntimeInstruction::Branch(
                    label_of(&target)?,
                ));
            }
            Fixup::ConditionalBranch {
                at,
                on_true,
                on_false,
            } => {
                code[at] =
                    MachineInstruction::Cpu(vihaco_cpu::RuntimeInstruction::ConditionalBranch(
                        label_of(&on_true)?,
                        label_of(&on_false)?,
                    ));
            }
            Fixup::Call { at, callee } => {
                let target = function_of(&callee)?;
                let arity = match &code[at] {
                    MachineInstruction::Cpu(vihaco_cpu::RuntimeInstruction::Call(a, _)) => *a,
                    _ => unreachable!("a Call fixup always points at a Call"),
                };
                code[at] =
                    MachineInstruction::Cpu(vihaco_cpu::RuntimeInstruction::Call(arity, target));
            }
        }
    }

    if main_function.is_none() {
        return Err(ResolveError::MissingMain);
    }

    // `LocalModule` is a foreign type, so it is built field by field rather
    // than with a struct literal.
    #[allow(clippy::field_reassign_with_default)]
    let module = {
        let mut module = Program::default();
        module.code = code;
        module.functions = function_infos;
        module.labels = labels;
        module.strings = interner.strings;
        module.main_function = main_function;
        module.extra = extra;
        module
    };
    Ok(module)
}
