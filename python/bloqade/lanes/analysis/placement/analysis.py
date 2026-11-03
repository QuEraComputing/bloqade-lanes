from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bloqade.analysis.address.lattice import Address, AddressQubit
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from kirin.interp.exceptions import InterpreterError
from typing_extensions import Self

from bloqade.lanes.analysis.code_blocks import (
    CodeBlock,
    LocalCodeBlock,
    localize_code_blocks,
)
from bloqade.lanes.bytecode.encoding import LocationAddress

from .lattice import AtomState, ConcreteState
from .strategy import PlacementStrategyABC

if TYPE_CHECKING:
    from bloqade.lanes.dialects.place import CZ


@dataclass
class PlacementAnalysis(Forward[AtomState]):
    keys = ("runtime.placement",)

    initial_layout: tuple[LocationAddress, ...]
    address_analysis: dict[ir.SSAValue, Address]
    move_count: defaultdict[ir.SSAValue, int] = field(init=False)
    cz_lookahead_buffers: dict[
        ir.Block, tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    ] = field(default_factory=dict, init=False, repr=False)
    cz_lookahead_stmt_positions: dict[ir.Block, dict["CZ", int]] = field(
        default_factory=dict, init=False, repr=False
    )

    placement_strategy: PlacementStrategyABC
    """The strategy function to use for calculating placements."""
    code_blocks: tuple[CodeBlock, ...] = field(default=(), kw_only=True)
    """Code blocks honored by the initial layout, in global qubit ids."""
    _local_code_blocks: tuple[LocalCodeBlock, ...] = field(
        default=(), init=False, repr=False
    )
    lattice = AtomState

    def __post_init__(self):
        self.placement_strategy.validate_initial_layout(self.initial_layout)
        super().__post_init__()

    def initialize(self) -> Self:
        self.move_count = defaultdict(int)
        self.cz_lookahead_buffers.clear()
        self.cz_lookahead_stmt_positions.clear()
        return super().initialize()

    def get_inintial_state(self, qubits: tuple[ir.SSAValue, ...]):
        occupied = set(self.initial_layout)
        layout = []
        move_count = []
        qids = []
        for q in qubits:
            if not isinstance(addr := self.address_analysis.get(q), AddressQubit):
                raise InterpreterError(f"Qubit {q} does not have a qubit address.")

            qids.append(addr.data)
            loc_addr = self.initial_layout[addr.data]
            occupied.discard(loc_addr)
            layout.append(loc_addr)
            move_count.append(self.move_count[q])

        self._local_code_blocks = localize_code_blocks(self.code_blocks, qids)
        return ConcreteState(
            layout=tuple(layout),
            occupied=frozenset(occupied),
            move_count=tuple(move_count),
            code_blocks=self._local_code_blocks,
        )

    def attach_code_blocks(self, state: AtomState) -> AtomState:
        """Set the current static circuit's code blocks on ``state``.

        Called on the state passed to, and every state returned from, a
        placement strategy, so strategies that construct a ``ConcreteState``
        without the field still hand it on.
        """
        if (
            self._local_code_blocks
            and isinstance(state, ConcreteState)
            and state.code_blocks is not self._local_code_blocks
        ):
            state.code_blocks = self._local_code_blocks
        return state

    def build_cz_buffer(
        self, block: ir.Block
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        from bloqade.lanes.dialects.place import CZ

        buffer: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        stmt_positions: dict[CZ, int] = {}
        for node in block.stmts:
            if isinstance(node, CZ):
                stmt_positions[node] = len(buffer)
                buffer.append((node.controls, node.targets))
        self.cz_lookahead_stmt_positions[block] = stmt_positions
        return tuple(buffer)

    def buffered_future_cz_layers(
        self, stmt: "CZ"
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        if stmt.parent_block is None:
            return ()
        block = stmt.parent_block
        assert (
            block in self.cz_lookahead_buffers
        ), "Expected CZ lookahead buffer for block to be initialized before CZ execution"
        assert (
            block in self.cz_lookahead_stmt_positions
        ), "Expected CZ lookahead statement positions for block to be initialized before CZ execution"
        buffer = self.cz_lookahead_buffers[block]
        stmt_positions = self.cz_lookahead_stmt_positions[block]

        current_layer = (stmt.controls, stmt.targets)
        stmt_position = stmt_positions.get(stmt)
        if stmt_position is None:
            raise InterpreterError(
                "Lookahead CZ buffer missing current CZ statement mapping"
            )
        if stmt_position >= len(buffer) or buffer[stmt_position] != current_layer:
            raise InterpreterError(
                "Lookahead CZ buffer out of sync with executed CZ order"
            )
        return buffer[stmt_position:]

    def method_self(self, method: ir.Method) -> AtomState:
        return AtomState.bottom()

    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        return tuple(AtomState.bottom() for _ in node.results)
