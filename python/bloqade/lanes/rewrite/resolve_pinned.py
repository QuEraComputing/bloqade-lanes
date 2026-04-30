from dataclasses import dataclass

from bloqade.analysis import address
from kirin import ir
from kirin.rewrite import abc

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import place


@dataclass
class ResolvePinnedAddresses(abc.RewriteRule):
    """Stamp each NewLogicalQubit's location_address from the analysis output.

    For NewLogicalQubits that already have a non-None location_address (i.e.
    user-pinned), the attribute is left alone — the heuristic respected it
    and the layout entry should match.

    For NewLogicalQubits with location_address=None, the heuristic's choice is
    looked up via address_entries[stmt.result] -> AddressQubit.data, which
    indexes into initial_layout.

    Post-condition: every NewLogicalQubit has a non-None location_address.
    """

    address_entries: dict[ir.SSAValue, address.Address]
    initial_layout: tuple[LocationAddress, ...]

    def rewrite_Statement(self, node: ir.Statement) -> abc.RewriteResult:
        if not isinstance(node, place.NewPinnedQubit):
            return abc.RewriteResult()
        if node.location_address is not None:
            return abc.RewriteResult()
        addr_entry = self.address_entries.get(node.result)
        if not isinstance(addr_entry, address.AddressQubit):
            return abc.RewriteResult()
        if addr_entry.data >= len(self.initial_layout):
            return abc.RewriteResult()
        node.location_address = self.initial_layout[addr_entry.data]
        return abc.RewriteResult(has_done_something=True)
