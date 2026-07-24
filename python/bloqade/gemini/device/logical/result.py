from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from bloqade.core.device import Result

from bloqade.gemini import logical
from bloqade.gemini.post_processing import generate_post_processing

RetType = TypeVar("RetType")


@dataclass(kw_only=True)
class GeminiLogicalResult(Result):
    """Result view over stored Gemini logical shots.

    Merge-oriented methods assume each selected task ID has the same subtask
    structure. Post-processing is applied to each selected subtask's flat shot
    array.

    Attributes:
        storage (StorageBackend): Storage backend that holds shots and task
            metadata.
        shot_filter (ShotFilter): Filter used when reading shots and deriving
            subtask scope. Defaults to the DETECTED frame type.
    """

    def postprocessing_functions(self) -> dict[int, Callable | None]:
        """Decode stored programs and build post-processing functions.

        Program records are scoped by `shot_filter.task_ids`. When multiple
        task IDs share a `program_index`, the first stored program at that index
        is used.

        Returns:
            dict[int, Callable | None]: Mapping from program index to its
                generated post-processing function.
        """
        task_ids = self.shot_filter.task_ids
        programs = self.storage.get_programs(task_ids=task_ids)
        postprocessing_functions = {}
        for program in programs:
            idx = program["program_index"]
            if idx in postprocessing_functions:
                # NOTE: merging across task_ids means we assume all of them identical
                continue
            kernel_json = program["content"]
            kernel_mt = logical.kernel.decode_json(kernel_json)
            postprocessing_function = generate_post_processing(kernel_mt)
            postprocessing_functions[idx] = postprocessing_function

        return postprocessing_functions

    def logical_results(
        self,
        verify: bool = True,
        postprocessing_functions: (
            dict[int, Callable[[np.ndarray], RetType] | None] | None
        ) = None,
    ) -> list[RetType | np.ndarray]:
        """Return logical results grouped by merged subtask.

        Args:
            verify (bool): Whether to validate that selected task IDs can be
                merged before reading shots. Defaults to True.
            postprocessing_functions (dict[int, Callable | None] | None):
                Optional mapping from program index to post-processing function.
                When None, functions are built from stored programs. Defaults
                to None.

        Returns:
            list[RetType]: Post-processed results for each merged subtask.
                If a post-processing function is None, the physical shot array is
                returned for that subtask.

        Raises:
            ValueError: If `verify` is True and selected task IDs cannot be
                merged.
        """
        ret_vals = []
        subtasks = self.subtasks(verify=verify)
        if postprocessing_functions is None:
            postprocessing_functions = self.postprocessing_functions()
        shot_results = self._shot_results_for_subtasks(subtasks)

        for shot_result, subtask in zip(shot_results, subtasks):
            func = postprocessing_functions[subtask["program_index"]]
            if func is None:
                ret_vals.append(shot_result)
            else:
                ret_vals.append(func(shot_result))

        return ret_vals
