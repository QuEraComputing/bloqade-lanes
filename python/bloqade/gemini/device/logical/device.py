from dataclasses import dataclass, field
from importlib.metadata import version

from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.core.device import Device
from bloqade.core.device.task import (
    KernelBatchTask,
    ParameterScanTask,
    SingleKernelTask,
)
from kirin import ir
from kirin.validation import ValidationSuite

from bloqade.gemini.logical.group import kernel as logical_kernel

from ...common.validation.duplicate_address import (
    DuplicateAddressValidation,
)
from ...logical.validation.clifford.analysis import (
    GeminiLogicalValidation,
)
from ...logical.validation.measurement.analysis import (
    GeminiTerminalMeasurementValidation,
)
from .future import GeminiLogicalFuture
from .task import GeminiKernelBatchTask, GeminiParameterScanTask, GeminiSingleKernelTask

_bloqade_version = version("bloqade-circuit")


@dataclass(kw_only=True)
class GeminiLogicalDevice(Device[GeminiLogicalFuture]):
    """Device that builds tasks for the Gemini logical backend.

    Wires the Gemini-specific task and future classes into the generic
    `Device` factory and defaults `context_name` to "gemini-logical".
    """

    program_language: str = "squin"
    language_version: str = "0.1.0"
    validation_suite: ValidationSuite | None = field(
        default_factory=lambda: ValidationSuite(
            [
                GeminiLogicalValidation,
                GeminiTerminalMeasurementValidation,
                FlatKernelNoCloningValidation,
                DuplicateAddressValidation,
            ]
        )
    )
    dialect_group: ir.DialectGroup = field(default_factory=lambda: logical_kernel)

    future_cls: type[GeminiLogicalFuture] = GeminiLogicalFuture
    single_kernel_task_cls: type[SingleKernelTask[GeminiLogicalFuture]] = field(
        default=GeminiSingleKernelTask,
        init=False,
    )
    parameter_scan_task_cls: type[ParameterScanTask[GeminiLogicalFuture]] = field(
        default=GeminiParameterScanTask,
        init=False,
    )
    kernel_batch_task_cls: type[KernelBatchTask[GeminiLogicalFuture]] = field(
        default=GeminiKernelBatchTask,
        init=False,
    )

    context_name: str = "gemini-logical"
