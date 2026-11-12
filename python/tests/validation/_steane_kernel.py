"""Build validation fixtures from lanes dialects without importing Gemini."""

from bloqade.rewrite.passes import AggressiveUnroll
from bloqade.squin import gate, qubit
from kirin import ir, rewrite
from kirin.passes import Default
from kirin.prelude import structural_no_opt

from bloqade.lanes.dialects import arch, steane713
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.rewrite.qubit_count import InsertQubitCount


@ir.dialect_group(structural_no_opt.union([gate, qubit, arch, steane713, extensions]))
def kernel(self):
    def run_pass(mt, *, unroll: bool = True):
        if unroll:
            AggressiveUnroll(self).fixpoint(mt)
        else:
            Default(self, verify=False).fixpoint(mt)
        rewrite.Walk(InsertQubitCount(7)).rewrite(mt.code)

    return run_pass
