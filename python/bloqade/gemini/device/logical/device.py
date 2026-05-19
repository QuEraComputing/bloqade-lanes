from dataclasses import dataclass, field

from bloqade.core.device import Device
from bloqade.core.device.task import (
    KernelBatchTask,
    ParameterScanTask,
    SingleKernelTask,
)

from .future import GeminiLogicalFuture
from .task import GeminiKernelBatchTask, GeminiParameterScanTask, GeminiSingleKernelTask


@dataclass(kw_only=True)
class GeminiLogicalDevice(Device[GeminiLogicalFuture]):
    """Device that builds tasks for the Gemini logical backend.

    Wires the Gemini-specific task and future classes into the generic
    `Device` factory and defaults `context_name` to "gemini-logical".
    """

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
