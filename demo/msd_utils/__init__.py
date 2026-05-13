"""Utilities shared by the magic-state distillation demo notebooks."""

from .application.baselines import (
    infer_distilled_sign_vector,
    infer_factory_target,
    injected_baseline,
    naive_distilled_summary,
    naive_injected_summary,
)
from .application.constants import (
    DEFAULT_BASIS_LABELS,
    DEFAULT_IDEAL_FACTORY_ACCEPTANCE,
    DEFAULT_TARGET_BLOCH,
)
from .application.mld import (
    build_mld_decoders_from_pair,
    estimate_mld_ancilla_scores,
    estimate_mld_ancilla_scores_from_tasks,
    train_mld_decoder_pair,
    train_mld_decoder_pair_from_task,
)
from .application.mle import build_mle_decoders
from .application.msd_kernels import (
    DecoderKernelBundle,
    build_decoder_kernel_bundle,
    build_injected_decoder_kernel_map,
    build_msd_primitives,
)
from .application.table_decoders import SparseTableDecoder, TableDecoderClass
from .application.thresholds import DecoderAdapter, evaluate_curve, evaluate_mld_curve
from .domain.confidence import TableDecoderWithConfidence
from .domain.kernels import DecoderPrimitiveSet, produce_tomography_kernels
from .domain.layout import DEFAULT_SYNDROME_LAYOUT, SyndromeLayout, split_factory_bits
from .domain.special_tasks import apply_special_tsim_circuit_strategy, build_task_map
from .domain.tasks import DemoTask
from .standard.bit_packing import (
    pack_boolean_array,
    packed_bits_to_int,
    unpack_packed_bits,
)
from .standard.measurement_maps import build_measurement_maps
from .standard.sampling import BasisDataset, run_task
from .standard.tomography import (
    expectation_conf_interval,
    expectation_with_error_bar,
    fidelity_from_counts,
    fidelity_from_zero_one_counts,
    logical_expectation,
    posterior_fidelity_summary,
)
from .standard.types import (
    DetectorErrorModelTask,
    DetectorObservableResult,
    FidelitySummary,
    KirinKernel,
    MeasurementMap,
    PosteriorFidelitySummary,
    SimulatorTask,
    SquinKernel,
    TsimCircuit,
)

__all__ = [
    "BasisDataset",
    "DEFAULT_BASIS_LABELS",
    "DEFAULT_IDEAL_FACTORY_ACCEPTANCE",
    "DEFAULT_SYNDROME_LAYOUT",
    "DEFAULT_TARGET_BLOCH",
    "DecoderAdapter",
    "DecoderKernelBundle",
    "DecoderPrimitiveSet",
    "DemoTask",
    "DetectorErrorModelTask",
    "DetectorObservableResult",
    "FidelitySummary",
    "KirinKernel",
    "MeasurementMap",
    "PosteriorFidelitySummary",
    "SimulatorTask",
    "SparseTableDecoder",
    "SquinKernel",
    "SyndromeLayout",
    "TableDecoderClass",
    "TableDecoderWithConfidence",
    "TsimCircuit",
    "apply_special_tsim_circuit_strategy",
    "build_decoder_kernel_bundle",
    "build_injected_decoder_kernel_map",
    "build_measurement_maps",
    "build_mld_decoders_from_pair",
    "build_mle_decoders",
    "build_msd_primitives",
    "build_task_map",
    "estimate_mld_ancilla_scores",
    "estimate_mld_ancilla_scores_from_tasks",
    "evaluate_curve",
    "evaluate_mld_curve",
    "expectation_conf_interval",
    "expectation_with_error_bar",
    "fidelity_from_counts",
    "fidelity_from_zero_one_counts",
    "infer_distilled_sign_vector",
    "infer_factory_target",
    "injected_baseline",
    "logical_expectation",
    "naive_distilled_summary",
    "naive_injected_summary",
    "pack_boolean_array",
    "packed_bits_to_int",
    "posterior_fidelity_summary",
    "produce_tomography_kernels",
    "run_task",
    "split_factory_bits",
    "train_mld_decoder_pair",
    "train_mld_decoder_pair_from_task",
    "unpack_packed_bits",
]
