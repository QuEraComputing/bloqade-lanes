from dataclasses import dataclass
from typing import Literal

from bloqade.rewrite.passes import aggressive_unroll as agg
from bloqade.squin.rewrite import SquinU3ToClifford
from kirin import ir, rewrite
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.lanes import layout
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite import move2squin
from bloqade.lanes.rewrite.move2squin import (
    NoiseModelABC as NoiseModelABC,
    SimpleNoiseModel as SimpleNoiseModel,
)


@dataclass
class MoveToSquinTransformer:
    arch_spec: layout.ArchSpec
    logical_initialization: ir.Method[
        [float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None
    ]
    noise_model: NoiseModelABC | None = None
    aggressive_unroll: bool = False

    def transform(self, main: ir.Method) -> ir.Method:
        main = main.similar(main.dialects.union(squin.kernel))

        interp = atom.AtomInterpreter(main.dialects, arch_spec=self.arch_spec)
        frame, _ = interp.run(main)

        rule = move2squin.InsertQubits()
        rewrite.Walk(rule).rewrite(main.code)

        if self.noise_model is None:
            rule = move2squin.InsertGates(
                self.arch_spec,
                tuple(rule.physical_ssa_values),
                frame.atom_state_map,
                self.logical_initialization,
            )
        else:
            rule = rewrite.Chain(
                move2squin.InsertGates(
                    self.arch_spec,
                    tuple(rule.physical_ssa_values),
                    frame.atom_state_map,
                    self.logical_initialization,
                ),
                move2squin.InsertNoise(
                    self.arch_spec,
                    tuple(rule.physical_ssa_values),
                    frame.atom_state_map,
                    self.noise_model,
                ),
            )

        rewrite.Walk(rule).rewrite(main.code)
        if self.aggressive_unroll:
            agg.AggressiveUnroll(main.dialects).fixpoint(main)
        else:
            agg.Fold(main.dialects)(main)

        rewrite.Walk(
            rewrite.Chain(
                SquinU3ToClifford(),
                move2squin.CleanUpMoveDialect(frame.atom_state_map),
            )
        ).rewrite(main.code)

        out = main.similar(main.dialects.discard(move.dialect))
        out.verify()

        return out
