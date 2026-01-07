from dataclasses import dataclass
from typing import Literal

from bloqade.rewrite.passes import aggressive_unroll as agg
from bloqade.squin.rewrite import SquinU3ToClifford
from kirin import ir, rewrite
from kirin.dialects import ilist
from kirin.passes import TypeInfer

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
class MoveToSquin:
    arch_spec: layout.ArchSpec
    logical_initialization: ir.Method[
        [float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None
    ]
    noise_model: NoiseModelABC | None = None
    aggressive_unroll: bool = False

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        main = main.similar(main.dialects.union(squin.kernel))

        vqpu = atom.AtomInterpreter(main.dialects, arch_spec=self.arch_spec)
        run_method = vqpu.run_no_raise if no_raise else vqpu.run

        frame, _ = run_method(main)
        qubit_rule = move2squin.InsertQubits(frame)
        rewrite.Walk(qubit_rule).rewrite(main.code)
        rules = [
            move2squin.InsertGates(
                self.arch_spec,
                qubit_rule.physical_ssa_values,
                frame,
                self.logical_initialization,
            ),
            move2squin.InsertMeasurements(qubit_rule.physical_ssa_values, frame),
        ]
        if self.noise_model is not None:
            rules.append(
                move2squin.InsertNoise(
                    self.arch_spec,
                    qubit_rule.physical_ssa_values,
                    frame,
                    self.noise_model,
                ),
            )

        rewrite.Walk(rewrite.Chain(*rules)).rewrite(main.code)

        # we need to fold before writing U3 to Clifford
        if self.aggressive_unroll:
            agg.AggressiveUnroll(main.dialects).fixpoint(main)
        else:
            agg.Fold(main.dialects)(main)

        rewrite.Walk(
            SquinU3ToClifford(),
        ).rewrite(main.code)

        rewrite.Fixpoint(rewrite.Walk(move2squin.CleanUpMoveDialect())).rewrite(
            main.code
        )

        out = main.similar(main.dialects.discard(move.dialect))

        TypeInfer(out.dialects)(out)
        out.verify()
        out.verify_type()

        return out
