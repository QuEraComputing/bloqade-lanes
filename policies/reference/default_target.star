# Reference Target Generator DSL policy.
#
# Mirrors the in-tree `DefaultTargetGenerator`: every target qubit stays put
# and each control qubit moves to the CZ blockade partner of its target's
# location.
#
# Policy contract (Plan B of #597):
#   def generate(ctx, lib) -> list[dict[int, Location]]
#
# `ctx` exposes:
#   - ctx.arch_spec           : ArchSpec wrapper (read-only)
#   - ctx.placement           : Placement (dict-like; .qubits(), .get(qid),
#                               .items(), .len)
#   - ctx.controls            : list[int]
#   - ctx.targets             : list[int]
#   - ctx.lookahead_cz_layers : list[(list[int], list[int])]
#   - ctx.cz_stage_index      : int
#
# `lib` exposes:
#   - lib.arch_spec           : ArchSpec wrapper (alias of ctx.arch_spec)
#   - lib.cz_partner(loc)     : Location | None
#
# Determinism: the kernel iterates ctx.placement in qid order; both the
# returned dict and the validation step preserve that ordering.

def generate(ctx, lib):
    target = {}
    for q in ctx.placement.qubits():
        target[q] = ctx.placement.get(q)
    for i in range(len(ctx.controls)):
        c = ctx.controls[i]
        t = ctx.targets[i]
        partner = lib.cz_partner(target[t])
        if partner == None:
            return []   # defer to fallback DefaultTargetGenerator
        target[c] = partner
    return [target]
