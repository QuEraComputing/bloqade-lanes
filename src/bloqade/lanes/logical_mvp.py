import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import stim
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.rewrite.initialize import __RewriteU3ToInitialize
from bloqade.native.upstream import SquinToNative
from bloqade.rewrite.passes import CallGraphPass
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim
from kirin import ir, passes, rewrite
from kirin.dialects import ilist
from tsim import Circuit

from bloqade import annotate, squin, types
from bloqade.lanes import visualize
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite import transversal
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.upstream import NativeToPlace, PlaceToMove

# Kernel definition with gemini logical dialect and annotate
kernel = squin.kernel.add(gemini_logical.dialect).add(annotate)
kernel.run_pass = squin.kernel.run_pass


def simulate_program(program: Circuit, shots: int):
    sampler = program.compile_detector_sampler()
    return sampler.sample(shots, separate_observables=True)


@dataclass
class JobResult:
    measurement_bits: np.ndarray
    detector_bits: np.ndarray
    logical_bits: np.ndarray
    detector_error_model: stim.DetectorErrorModel
    phyiscal_program: Circuit


class Job:
    def __init__(self, stim_program_str: str, shots: int):
        program = Circuit(stim_program_str)
        dem = program.detector_error_model(approximate_disjoint_errors=True)
        sampler = program.compile_sampler()
        samples = sampler.sample(shots)

        m2dconverter = program._stim_circ.compile_m2d_converter(
            skip_reference_sample=True
        )
        dets, obs = m2dconverter.convert(
            measurements=samples, separate_observables=True
        )

        self._result = JobResult(
            measurement_bits=samples,
            detector_error_model=dem,
            detector_bits=dets,
            logical_bits=obs,
            phyiscal_program=program,
        )

    def get_results(self) -> JobResult:
        return self._result


class GeminiLogical:
    def submit(
        self,
        method,
        shots: int = 100,
        m2obs: np.ndarray | None = None,
        m2dets: np.ndarray | None = None,
    ) -> Job:
        stim_program_str = str(compile_to_physical_stim_program(method))

        if m2obs is not None:
            num_meas = m2obs.shape[0]
            targets = " ".join(str(i) for i in range(num_meas))
            stim_program_str += f"\nM {targets}\n"

            for i in range(m2obs.shape[1]):
                recs = np.flatnonzero(m2obs[:, 0]) - num_meas
                rec_str = " ".join(f"rec[{rec}]" for rec in recs)
                stim_program_str += f"OBSERVABLE_INCLUDE({i}) {rec_str}\n"
        if m2dets is not None:
            for i in range(m2dets.shape[1]):
                recs = np.flatnonzero(m2dets[:, 0]) - m2dets.shape[0]
                rec_str = " ".join(f"rec[{rec}]" for rec in recs)
                stim_program_str += f"DETECTOR {rec_str}\n"
        return Job(stim_program_str, shots)


class BaseDecoder:
    def __init__(
        self,
        dem: stim.DetectorErrorModel,
    ):
        pass

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        """Decode a single shot of detector bits."""
        pass

    def decode(self, detector_bits: np.ndarray, logical_bits: np.ndarray | None = None) -> np.ndarray:
        """Decode a batch or single shot of detector bits."""
        if detector_bits.ndim == 1:
            if logical_bits is not None:
                return self._decode(detector_bits) ^ logical_bits
            return self._decode(detector_bits)
        else:
            res = []
            for i in range(detector_bits.shape[0]):
                if logical_bits is not None:
                    res.append(self._decode(detector_bits[i]) ^ logical_bits[i])
                else:
                    res.append(self._decode(detector_bits[i]))
            return np.array(res)

    def decode_and_return_soft_information(
        self, detector_bits: np.ndarray
    ) -> np.ndarray:
        """Decode a batch or single shot of detector bits and return soft information."""
        pass


class TesseractDecoder(BaseDecoder):
    def __init__(self, dem: stim.DetectorErrorModel):
        from tesseract_decoder import tesseract

        config = tesseract.TesseractConfig(dem=dem)
        self.num_observables = dem.num_observables
        self._decoder = config.compile_decoder()

    def _decode(self, detector_bits: np.ndarray) -> np.ndarray:
        return self._decoder.decode(detector_bits)


@kernel
def set_detector(meas: ilist.IList[ilist.IList[types.MeasurementResult, Any], Any]):
    h = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 4, 6]]

    num_meas = len(meas)

    for i in range(len(h)):
        stab = h[i]
        res = []
        for j in range(4):
            for k in range(num_meas):
                m = meas[k]
                res = res + [m[stab[j]]]

        annotate.set_detector(res, coordinates=[0, i])


@kernel
def set_observable(meas: ilist.IList[types.MeasurementResult, Any], idx: int):
    annotate.set_observable([meas[0], meas[1], meas[5]], idx)


def transversal_rewrites(mt: ir.Method):
    """Apply transversal rewrite rules to a squin method.

    Args:
        mt (ir.Method): rewrite the method in place.

    Returns:
        ir.Method: The rewritten method.

    """

    rewrite.Walk(
        rewrite.Chain(
            transversal.RewriteLocations(logical.steane7_transversal_map),
            transversal.RewriteLogicalInitialize(logical.steane7_transversal_map),
            transversal.RewriteMoves(logical.steane7_transversal_map),
            transversal.RewriteGetItem(logical.steane7_transversal_map),
            transversal.RewriteLogicalToPhysicalConversion(),
        )
    ).rewrite(mt.code)

    return mt


def compile_squin_to_move(mt: ir.Method, transversal_rewrite: bool = False):
    """Compile a squin kernel to move dialect.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False
    Returns:
        ir.Method: The compiled move dialect method.
    """

    # Compile to move dialect
    rule = rewrite.Chain(
        rewrite.Walk(
            RewriteNonCliffordToU3(),
        ),
        rewrite.Walk(
            __RewriteU3ToInitialize(),
        ),
    )

    CallGraphPass(mt.dialects, rule)(mt)
    mt = SquinToNative().emit(mt)
    mt = NativeToPlace().emit(mt)

    mt = PlaceToMove(
        fixed.LogicalLayoutHeuristic(),
        fixed.LogicalPlacementStrategy(),
        fixed.LogicalMoveScheduler(),
    ).emit(mt)
    if transversal_rewrite:
        mt = transversal_rewrites(mt)

    passes.TypeInfer(mt.dialects)(mt)

    mt.verify()
    mt.verify_type()

    return mt


def compile_squin_to_move_and_visualize(
    mt: ir.Method, interactive: bool = True, transversal_rewrite: bool = False
):
    """Compile a squin kernel to moves and visualize the program.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        interactive (bool, optional): Whether to display the visualization interactively. Defaults to True.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
    """
    # Compile to move dialect
    mt = compile_squin_to_move(mt, transversal_rewrite)
    if transversal_rewrite:
        arch_spec = generate_arch(4)
        marker = "o"
    else:
        arch_spec = logical.get_arch_spec()
        marker = "s"

    visualize.debugger(mt, arch_spec, interactive=interactive, atom_marker=marker)


def compile_to_physical_squin_noise_model(
    mt: ir.Method, noise_model: NoiseModelABC | None = None
) -> ir.Method:
    """Compiles a logical squin kernel to a physical squin kernel with noise channels inserted.

    Args:
        mt: The logical squin method to compile.
        noise_model: The noise model to insert during compilation.
    Returns:
        The compiled physical squin method.

    """
    if noise_model is None:
        noise_model = generate_simple_noise_model()

    move_mt = compile_squin_to_move(mt, transversal_rewrite=True)
    transformer = MoveToSquin(
        arch_spec=generate_arch(4),
        logical_initialization=logical.steane7_initialize,
        noise_model=noise_model,
        aggressive_unroll=False,
    )

    return transformer.emit(move_mt)


def compile_to_physical_stim_program(
    mt: ir.Method, noise_model: NoiseModelABC | None = None
) -> Circuit:
    """Compiles a logical squin kernel to a physical stim kernel with noise channels inserted.

    Args:
        mt: The logical squin method to compile.
        noise_model: The noise model to insert during compilation.
    Returns:
        The compiled physical stim program as a string.

    """
    from tsim import Circuit

    noise_kernel = compile_to_physical_squin_noise_model(mt, noise_model)
    noise_kernel = squin_to_stim(noise_kernel)

    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)

    return Circuit(buf.getvalue().strip())
