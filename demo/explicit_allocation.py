"""Demo: explicit qubit allocation.

Pin two logical qubits to known physical addresses with
``gemini.common.new_at`` and let the layout heuristic place the rest. The
compiled move IR's ``move.Fill`` carries the requested addresses for the
pinned qubits and heuristic-chosen addresses for the un-pinned qubits.
"""

from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.common.dialects.qubit import new_at
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.upstream import squin_to_move


@gemini_logical.kernel(aggressive_unroll=True)
def main():
    # Pinned qubits at explicit physical addresses.
    a = new_at(0, 0, 0)
    b = new_at(0, 4, 0)
    # Un-pinned qubits — the layout heuristic chooses their home sites.
    reg = squin.qalloc(2)
    # CZ between pinned and un-pinned qubits.
    squin.broadcast.cz(ilist.IList([a]), ilist.IList([reg[0]]))
    squin.broadcast.cz(ilist.IList([b]), ilist.IList([reg[1]]))
    gemini_logical.terminal_measure(ilist.IList([a, b, reg[0], reg[1]]))


compiled = squin_to_move(
    main,
    layout_heuristic=LogicalLayoutHeuristic(),
    placement_strategy=LogicalPlacementStrategyNoHome(),
)

fills = [s for s in compiled.callable_region.walk() if isinstance(s, move.Fill)]
assert len(fills) == 1
print("move.Fill location_addresses (pinned + heuristic-chosen):")
for addr in fills[0].location_addresses:
    print(f"  zone={addr.zone_id} word={addr.word_id} site={addr.site_id}")
