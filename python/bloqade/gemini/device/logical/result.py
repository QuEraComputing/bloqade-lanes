from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar, overload

import numpy as np
from bloqade.core.device import Result

from bloqade.gemini import logical
from bloqade.gemini.post_processing import generate_post_processing
from bloqade.lanes.analysis.atom import PostProcessing

from .utils import (
    aligned_detected_and_sorted_shots_for_subtasks,
    get_slm_mapping_postprocessing,
    shot_results_for_subtasks,
)

RetType = TypeVar("RetType")


@dataclass(kw_only=True)
class GeminiLogicalResult(Result, Generic[RetType]):
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

    @cached_property
    def _slm_postprocessing_functions(
        self,
    ) -> dict[
        int,
        tuple[Callable[[np.ndarray], list[list[bool]]], PostProcessing[RetType]],
    ]:
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
            kernel_mt = logical.kernel.decode_json(kernel_json)  # type: ignore[attr-defined]
            slm_to_raw, postprocessing_function = get_slm_mapping_postprocessing(
                kernel_mt
            )
            # NOTE: what if err in postproc fn generation/errors here??
            postprocessing_functions[idx] = (slm_to_raw, postprocessing_function)

        return postprocessing_functions

    def postprocessing_functions(self) -> dict[int, Callable | None]:
        """Build legacy compact-shot postprocessors from stored programs.

        This method preserves the original ``GeminiLogicalResult`` API for
        result stores whose bitstrings are already in compact physical
        measurement order. Raw 160-site SLM frames use
        ``_slm_postprocessing_functions`` instead.
        """
        task_ids = self.shot_filter.task_ids
        programs = self.storage.get_programs(task_ids=task_ids)
        postprocessing_functions = {}
        for program in programs:
            idx = program["program_index"]
            if idx in postprocessing_functions:
                continue
            kernel_mt = logical.kernel.decode_json(program["content"])  # type: ignore[attr-defined]
            postprocessing_functions[idx] = generate_post_processing(kernel_mt)

        return postprocessing_functions

    @cached_property
    def measurements(self) -> Sequence[Sequence[Sequence[bool]]]:
        ret_vals: list[list[list[bool]]] = []
        # TODO: OK to set verify=True?
        subtasks = self.subtasks(verify=True)
        postprocessing_functions = self._slm_postprocessing_functions
        shot_results = shot_results_for_subtasks(
            self.storage, self.shot_filter, subtasks, frame_type="DETECTED"
        )

        for shot_result, subtask in zip(shot_results, subtasks):
            func = postprocessing_functions[subtask["program_index"]][0]
            ret_vals.append(func(shot_result))

        return ret_vals

    @overload
    def logical_results(
        self,
        verify: bool = True,
        postprocessing_functions: None = None,
    ) -> list[list[RetType]]: ...

    @overload
    def logical_results(
        self,
        verify: bool,
        postprocessing_functions: dict[int, Callable[[np.ndarray], RetType] | None],
    ) -> list[RetType | np.ndarray]: ...

    def logical_results(
        self,
        verify: bool = True,
        postprocessing_functions: (
            dict[int, Callable[[np.ndarray], RetType] | None] | None
        ) = None,
    ) -> list[list[RetType]] | list[RetType | np.ndarray]:
        """Return logical kernel results grouped by merged subtask.

        When no postprocessor override is supplied, raw 160-site DETECTED
        frames are mapped into compact physical-measurement order and evaluated
        with the Atom postprocessor derived from each program. An explicitly
        supplied postprocessor retains the legacy behavior: it receives the
        raw DETECTED-frame array for its subtask.

        Args:
            verify: Validate that selected task IDs can be merged.
            postprocessing_functions: Optional legacy mapping from program
                index to a function accepting that subtask's raw shot array.
        """
        subtasks = self.subtasks(verify=verify)
        shot_results = shot_results_for_subtasks(
            self.storage,
            self.shot_filter,
            subtasks,
            frame_type="DETECTED",
        )

        if postprocessing_functions is not None:
            legacy_results: list[RetType | np.ndarray] = []
            for shot_result, subtask in zip(shot_results, subtasks):
                func = postprocessing_functions[subtask["program_index"]]
                legacy_results.append(
                    shot_result if func is None else func(shot_result)
                )
            return legacy_results

        is_raw_slm_frame = [
            shot_result.ndim == 2 and shot_result.shape[1] == 160
            for shot_result in shot_results
        ]
        if any(is_raw_slm_frame) and not all(is_raw_slm_frame):
            raise ValueError(
                "Cannot combine raw 160-site SLM frames with compact physical "
                "measurement frames in one result view."
            )

        if all(is_raw_slm_frame):
            generated_postprocessors = self._slm_postprocessing_functions
            return_values: list[list[RetType]] = []
            for shot_result, subtask in zip(shot_results, subtasks):
                slm_to_measurements, atom_postprocessor = generated_postprocessors[
                    subtask["program_index"]
                ]
                return_values.append(
                    list(
                        atom_postprocessor.emit_return(slm_to_measurements(shot_result))
                    )
                )
            return return_values

        legacy_postprocessors = self.postprocessing_functions()
        legacy_results: list[RetType | np.ndarray] = []
        for shot_result, subtask in zip(shot_results, subtasks):
            func = legacy_postprocessors[subtask["program_index"]]
            legacy_results.append(shot_result if func is None else func(shot_result))
        return legacy_results

    @cached_property
    def return_values(self) -> Sequence[Sequence[RetType]]:
        return [
            list(
                self._slm_postprocessing_functions[subtask["program_index"]][
                    1
                ].emit_return(measurements)
            )
            for measurements, subtask in zip(self.measurements, self.subtasks())
        ]

    @cached_property
    def detectors(self) -> Sequence[Sequence[Sequence[bool]]]:
        return [
            list(
                self._slm_postprocessing_functions[subtask["program_index"]][
                    1
                ].emit_detectors(measurements)
            )
            for measurements, subtask in zip(self.measurements, self.subtasks())
        ]

    @cached_property
    def observables(self) -> Sequence[Sequence[Sequence[bool]]]:
        return [
            list(
                self._slm_postprocessing_functions[subtask["program_index"]][
                    1
                ].emit_observables(measurements)
            )
            for measurements, subtask in zip(self.measurements, self.subtasks())
        ]

    @cached_property
    def filling_at_start(self) -> Sequence[Sequence[Sequence[bool]]]:
        ret_vals: list[list[list[bool]]] = []
        subtasks = self.subtasks(verify=True)
        postprocessing_functions = self._slm_postprocessing_functions
        _, shot_results = aligned_detected_and_sorted_shots_for_subtasks(
            self.storage,
            self.shot_filter,
            subtasks,
        )

        for shot_result, subtask in zip(shot_results, subtasks):
            func = postprocessing_functions[subtask["program_index"]][0]
            ret_vals.append(func(shot_result))

        # TOOD: can do validation later on sorted/detected frames
        return ret_vals
