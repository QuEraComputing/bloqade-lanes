"""Code-block membership carried from the kernel into layout and placement.

A code block is a group of physical qubits that together encode logical
information (for example one Steane or [[4,2,2]] block). The kernel registers
it with ``code_block.register``; ``resolve_code_blocks`` stamps a ``CodeBlockTag``
onto each member's ``place.NewPinnedQubit``; ``LayoutAnalysis`` groups the tags
into ``CodeBlock``s for the layout heuristic; ``PlacementAnalysis`` hands them to
placement strategies as ``LocalCodeBlock``s on ``ConcreteState``.

Every block's initial layout is a contiguous, position-ordered run of sites in a
single word: position ``p`` sits at site ``offset + p``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bloqade.lanes.bytecode.encoding import LocationAddress


@dataclass(frozen=True)
class CodeBlockTag:
    """Membership of one qubit: its block id and its position in the block."""

    block: int
    position: int


@dataclass(frozen=True)
class CodeBlock:
    """One code block in global qubit-id space."""

    block_id: int
    qids: tuple[int, ...]
    """Global qubit ids, in position order."""

    def __len__(self) -> int:
        return len(self.qids)


@dataclass(frozen=True)
class LocalCodeBlock:
    """One code block in a placement state's local index space."""

    block_id: int
    members: tuple[int | None, ...]
    """Local index of the qubit at each position, or None when that qubit is not
    an argument of the state."""


class CodeBlockWarning(UserWarning):
    """Registered code blocks were ignored."""


class CodeBlockPlacementError(ValueError):
    """The registered code blocks cannot be placed."""


def block_shape_error(
    locations: Sequence[LocationAddress], sites_per_word: int | None = None
) -> str | None:
    """Return why ``locations`` (in position order) break the block shape, or None.

    The shape: all in one word and zone, position ``p`` at site
    ``locations[0].site_id + p``, and, when ``sites_per_word`` is given, at most
    that many qubits.
    """
    if len(locations) == 0:
        return "the block is empty"
    if sites_per_word is not None and len(locations) > sites_per_word:
        return (
            f"the block has {len(locations)} qubits but a word has only "
            f"{sites_per_word} sites"
        )
    first = locations[0]
    words = {(loc.zone_id, loc.word_id) for loc in locations}
    if len(words) > 1:
        return f"the block spans several words: {sorted(words)}"
    for position, loc in enumerate(locations):
        if loc.site_id != first.site_id + position:
            sites = [loc.site_id for loc in locations]
            return (
                "position p must sit at site offset + p on contiguous sites, "
                f"got sites {sites}"
            )
    return None


def group_code_blocks(tags: dict[int, CodeBlockTag]) -> tuple[CodeBlock, ...]:
    """Group per-qubit tags into blocks, sorted by block id."""
    members: dict[int, dict[int, int]] = {}
    for qid, tag in tags.items():
        members.setdefault(tag.block, {})[tag.position] = qid
    blocks: list[CodeBlock] = []
    for block_id in sorted(members):
        by_position = members[block_id]
        if sorted(by_position) != list(range(len(by_position))):
            raise ValueError(
                f"code block {block_id} has non-contiguous positions "
                f"{sorted(by_position)}"
            )
        blocks.append(
            CodeBlock(
                block_id=block_id,
                qids=tuple(by_position[p] for p in range(len(by_position))),
            )
        )
    return tuple(blocks)


def localize_code_blocks(
    blocks: Sequence[CodeBlock], global_qids: Sequence[int]
) -> tuple[LocalCodeBlock, ...]:
    """Translate blocks into the local index space of ``global_qids``.

    Blocks with no member in ``global_qids`` are omitted.
    """
    local_index = {qid: idx for idx, qid in enumerate(global_qids)}
    local: list[LocalCodeBlock] = []
    for block in blocks:
        members = tuple(local_index.get(qid) for qid in block.qids)
        if all(m is None for m in members):
            continue
        local.append(LocalCodeBlock(block_id=block.block_id, members=members))
    return tuple(local)
