//! Observer trait for the Target Generator DSL kernel.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §5.2.
//!
//! Target generation is a single call per CZ stage: one `on_invoke`,
//! one `on_result` per stage. There is no per-step loop.

use serde::Serialize;
use std::io::Write;

#[derive(Debug, Clone, Serialize)]
pub struct TargetContextSnapshot {
    pub current_qubit_count: usize,
    pub controls_len: usize,
    pub targets_len: usize,
    pub lookahead_layers: usize,
    pub cz_stage_index: u32,
}

/// Summary of `TargetPolicyRunner::generate`'s `Vec<Vec<(u32, LocationAddr)>>`
/// result: `num_candidates` is the outer length; `first_candidate_size` is
/// the inner length for the first candidate (or 0 when empty).
#[derive(Debug, Clone, Serialize)]
pub struct CandidateSummary {
    pub num_candidates: usize,
    pub first_candidate_size: usize,
}

pub trait TargetKernelObserver {
    fn on_invoke(&mut self, _stage_idx: u64, _ctx: &TargetContextSnapshot) {}
    fn on_result(&mut self, _stage_idx: u64, _summary: &CandidateSummary, _ok: bool) {}
}

pub struct NoOpTargetObserver;
impl TargetKernelObserver for NoOpTargetObserver {}

const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
struct EnvInvoke<'a> {
    v: u32,
    kind: &'static str,
    stage: u64,
    ctx: &'a TargetContextSnapshot,
}
#[derive(Serialize)]
struct EnvResult<'a> {
    v: u32,
    kind: &'static str,
    stage: u64,
    summary: &'a CandidateSummary,
    ok: bool,
}

pub struct JsonTargetTraceObserver<W: Write> {
    writer: W,
}

impl<W: Write> JsonTargetTraceObserver<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
    fn emit<T: Serialize>(&mut self, env: &T) {
        let line = serde_json::to_string(env).expect("target trace serialization");
        let _ = writeln!(self.writer, "{line}");
        let _ = self.writer.flush();
    }
}

impl<W: Write> TargetKernelObserver for JsonTargetTraceObserver<W> {
    fn on_invoke(&mut self, stage: u64, ctx: &TargetContextSnapshot) {
        self.emit(&EnvInvoke {
            v: SCHEMA_VERSION,
            kind: "invoke",
            stage,
            ctx,
        });
    }
    fn on_result(&mut self, stage: u64, summary: &CandidateSummary, ok: bool) {
        self.emit(&EnvResult {
            v: SCHEMA_VERSION,
            kind: "result",
            stage,
            summary,
            ok,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_target_trace_observer_emits_invoke_and_result() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut obs = JsonTargetTraceObserver::new(&mut buf);
            obs.on_invoke(
                0,
                &TargetContextSnapshot {
                    current_qubit_count: 4,
                    controls_len: 1,
                    targets_len: 1,
                    lookahead_layers: 0,
                    cz_stage_index: 0,
                },
            );
            obs.on_result(
                0,
                &CandidateSummary {
                    num_candidates: 1,
                    first_candidate_size: 4,
                },
                true,
            );
        }
        let s = std::str::from_utf8(&buf).unwrap();
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains(r#""kind":"invoke""#));
        assert!(lines[1].contains(r#""kind":"result""#));
    }
}
