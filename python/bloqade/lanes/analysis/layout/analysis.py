import abc
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bloqade.analysis import address
from kirin import ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice import EmptyLattice

from bloqade.lanes.analysis.code_blocks import (
    CodeBlock,
    CodeBlockTag,
    CodeBlockWarning,
    block_shape_error,
    group_code_blocks,
)
from bloqade.lanes.bytecode.encoding import LocationAddress

if TYPE_CHECKING:
    from bloqade.lanes.arch.spec import ArchSpec


@dataclass
class LayoutHeuristicABC(abc.ABC):

    arch_spec: "ArchSpec"

    def _validate_pinned_in_arch(
        self,
        pinned: dict[int, LocationAddress],
        arch_spec: "ArchSpec",
    ) -> None:
        """Raise ValueError if any pinned address is not in the arch's home sites."""
        home_sites = arch_spec.home_sites
        bad = {addr for addr in pinned.values() if addr not in home_sites}
        if bad:
            sorted_bad = sorted(bad, key=lambda a: (a.word_id, a.site_id, a.zone_id))
            raise ValueError(
                f"pinned addresses are not valid home positions for this architecture: "
                f"{sorted_bad}. "
                f"Each pinned address must be one of the arch's home_sites."
            )

    @abc.abstractmethod
    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, LocationAddress] | None = None,
    ) -> tuple[LocationAddress, ...]:
        """
        Compute the initial qubit layout from circuit stages.

        Args:
            all_qubits: Tuple of logical qubit indices to be mapped.
            stages: List of circuit stages, where each stage is a tuple of
                (control, target) qubit pairs representing two-qubit gates.
            pinned: Map from logical qubit ID to pre-pinned LocationAddress.
                Implementations MUST place each pinned qubit at its requested
                address and MUST NOT use any address in pinned.values() for
                un-pinned qubits. None or empty preserves previous behavior.
                All values in pinned MUST be valid home positions for the
                architecture (i.e. present in arch_spec.home_sites); passing
                an out-of-arch address raises ValueError.

        Returns:
            A tuple of LocationAddress objects mapping logical qubit indices
            to physical locations. Pinned IDs return their pinned address;
            un-pinned IDs return the heuristic's choice. Raises if no legal
            layout exists.
        """
        ...  # pragma: no cover


class CodeBlockLayoutHeuristicABC(LayoutHeuristicABC):
    """A layout heuristic that can honor registered code blocks.

    ``LayoutAnalysis`` calls ``compute_layout_with_blocks`` instead of
    ``compute_layout`` whenever the kernel registers at least one block.
    """

    @abc.abstractmethod
    def compute_layout_with_blocks(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, LocationAddress],
        code_blocks: tuple[CodeBlock, ...],
    ) -> tuple[LocationAddress, ...]:
        """Compute an initial layout in which every block has the block shape.

        Each block in ``code_blocks`` must land on contiguous sites of a single
        word with position ``p`` at site ``offset + p``. Pinned qubits keep their
        pins; a fully pinned block already has the shape. Raises
        ``CodeBlockPlacementError`` when the blocks cannot all be placed.
        Contract otherwise as for ``compute_layout``.
        """
        ...  # pragma: no cover


@dataclass
class LayoutAnalysis(Forward):
    keys = ("place.layout",)
    lattice = EmptyLattice

    heuristic: LayoutHeuristicABC
    address_entries: dict[ir.SSAValue, address.Address]
    all_qubits: tuple[int, ...]
    stages: list[tuple[tuple[int, int], ...]] = field(default_factory=list, init=False)
    global_address_stack: list[int] = field(default_factory=list, init=False)
    location_addresses: dict[int, LocationAddress] = field(
        default_factory=dict, init=False
    )
    code_block_tags: dict[int, CodeBlockTag] = field(default_factory=dict, init=False)
    code_blocks: tuple[CodeBlock, ...] = field(default=(), init=False)
    """Blocks the last computed layout honors (empty if none were registered or
    the heuristic is not block-aware). Read after ``get_layout``."""

    def initialize(self):
        self.stages.clear()
        self.global_address_stack.clear()
        self.location_addresses.clear()
        self.code_block_tags.clear()
        self.code_blocks = ()
        return super().initialize()

    def eval_stmt_fallback(self, frame, stmt):
        return (self.lattice.bottom(),)

    def add_stage(self, control: tuple[int, ...], target: tuple[int, ...]):
        global_controls = tuple(self.global_address_stack[c] for c in control)
        global_targets = tuple(self.global_address_stack[t] for t in target)
        self.stages.append(tuple(zip(global_controls, global_targets)))

    def method_self(self, method: ir.Method):
        return EmptyLattice.bottom()

    def process_results(self):
        blocks = group_code_blocks(self.code_block_tags)
        self.code_blocks = ()
        if not blocks:
            return self.heuristic.compute_layout(
                self.all_qubits, self.stages, pinned=self.location_addresses
            )
        if not isinstance(self.heuristic, CodeBlockLayoutHeuristicABC):
            warnings.warn(
                f"layout heuristic {type(self.heuristic).__name__} is not "
                f"block-aware; ignoring {len(blocks)} registered code block(s).",
                CodeBlockWarning,
                stacklevel=2,
            )
            return self.heuristic.compute_layout(
                self.all_qubits, self.stages, pinned=self.location_addresses
            )
        layout = self.heuristic.compute_layout_with_blocks(
            self.all_qubits, self.stages, dict(self.location_addresses), blocks
        )
        sites_per_word = len(self.heuristic.arch_spec.words[0].site_indices)
        for block in blocks:
            reason = block_shape_error(
                [layout[qid] for qid in block.qids], sites_per_word
            )
            if reason is not None:
                raise RuntimeError(
                    f"{type(self.heuristic).__name__} broke code block "
                    f"{block.block_id}: {reason}"
                )
        self.code_blocks = blocks
        return layout

    def get_layout_no_raise(self, method: ir.Method):
        """Get the layout for a given method."""
        self.run_no_raise(method)
        return self.process_results()

    def get_layout(self, method: ir.Method):
        """Get the layout for a given method."""
        self.run(method)
        return self.process_results()

    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)
