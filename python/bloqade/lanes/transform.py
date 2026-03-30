from dataclasses import dataclass

from bloqade.rewrite.passes import aggressive_unroll as agg
from bloqade.squin.rewrite import SquinU3ToClifford
from kirin import ir, rewrite
from kirin.dialects import scf
from kirin.passes import TypeInfer

from bloqade import squin
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
    noise_model: NoiseModelABC
    add_noise: bool = False
    aggressive_unroll: bool = False

    def _resolve_initialize_kernel(self):
        """Resolve the clean initialization kernel for InsertGates."""
        clean, _ = self.noise_model.get_logical_initialize()
        return clean

    def _resolve_initialize_noise_kernel(self):
        """Resolve the noisy initialization kernel for InsertNoise.

        Only returns the noisy kernel when ``add_noise`` is True.
        """
        if not self.add_noise:
            return None
        _, noisy = self.noise_model.get_logical_initialize()
        return noisy

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        main = main.similar(main.dialects.union(squin.kernel.discard(scf.lowering)))

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
                self._resolve_initialize_kernel(),
            ),
            move2squin.InsertMeasurements(qubit_rule.physical_ssa_values, frame),
        ]
        if self.add_noise:
            rules.append(
                move2squin.InsertNoise(
                    self.arch_spec,
                    qubit_rule.physical_ssa_values,
                    frame,
                    self.noise_model,
                    initialize_noise_kernel=self._resolve_initialize_noise_kernel(),
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

        rewrite.Fixpoint(
            rewrite.Walk(
                rewrite.Chain(
                    move2squin.CleanUpMoveDialect(),
                    rewrite.DeadCodeElimination(),
                    rewrite.CommonSubexpressionElimination(),
                )
            )
        ).rewrite(main.code)

        out = main.similar(main.dialects.discard(move.dialect))

        TypeInfer(out.dialects)(out)
        out.verify()
        out.verify_type()

        return out
