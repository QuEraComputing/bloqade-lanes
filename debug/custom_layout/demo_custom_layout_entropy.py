"""Debug demo: entropy search from an explicit initial atom layout.

Run from the repository root:

    uv run python debug/custom_layout/demo_custom_layout_entropy.py

Use ``--no-interactive`` to capture and print the trace summary without opening
the matplotlib visualizer.
"""

from __future__ import annotations

import argparse

from bloqade import qubit, squin
from bloqade.lanes.bytecode._native import EntropyScorer
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.visualize.entropy_tree.controller import EntropyTreeController
from bloqade.lanes.visualize.entropy_tree.state import TreeStateReducer
from bloqade.lanes.visualize.entropy_tree.tracer import build_entropy_trace


@squin.kernel(typeinfer=True, fold=True)
def custom_layout_kernel():
    q = qubit.qalloc(4)

    # The CZ acts on q[0] and q[1]. q[2] and q[3] are spectator atoms that
    # block their starting sites during the traced entropy search.
    squin.cz(q[0], q[1])


CUSTOM_LAYOUT = {
    # q[0] and q[1] start in different home words, forcing entropy search to
    # synthesize movement before the CZ can execute.
    0: LocationAddress(word_id=0, site_id=0, zone_id=0),
    1: LocationAddress(word_id=4, site_id=7, zone_id=0),
    # Spectators: included because custom_layout is a full initial layout.
    2: LocationAddress(word_id=2, site_id=1, zone_id=0),
    3: LocationAddress(word_id=6, site_id=6, zone_id=0),
}


def _format_location(addr: LocationAddress) -> str:
    return f"zone={addr.zone_id} word={addr.word_id} site={addr.site_id}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize entropy search from a custom initial atom layout."
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--max-expansions", type=int, default=1000)
    parser.add_argument("--max-goal-candidates", type=int, default=3)
    parser.add_argument("--no-interactive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    max_expansions = None if args.max_expansions < 0 else args.max_expansions

    bundle = build_entropy_trace(
        kernel=custom_layout_kernel,
        kernel_name="custom_layout_kernel",
        layer_index=args.layer_index,
        max_expansions=max_expansions,
        max_goal_candidates=args.max_goal_candidates,
        custom_layout=CUSTOM_LAYOUT,
    )

    print("Custom initial layout:")
    for qid, addr in sorted(CUSTOM_LAYOUT.items()):
        print(f"  q[{qid}] -> {_format_location(addr)}")

    print("\nFirst traced target candidate:")
    for local_qid, addr in sorted(bundle.traced_target.items()):
        global_qid = bundle.local_to_global_qid.get(local_qid, local_qid)
        print(f"  q[{global_qid}] -> {_format_location(addr)}")

    print("\nBlocked spectator locations:")
    for addr in bundle.blocked_locations:
        global_qid = bundle.location_to_global_qid.get(addr)
        label = "unknown" if global_qid is None else f"q[{global_qid}]"
        print(f"  {label} at {_format_location(addr)}")

    print(
        f"\nCaptured {len(bundle.steps)} entropy-search steps "
        f"for CZ stage {args.layer_index}."
    )

    if args.no_interactive:
        return 0

    reducer = TreeStateReducer(
        steps=bundle.steps,
        root_node_id=bundle.root_node_id,
        best_buffer_size=bundle.best_buffer_size,
    )
    scorer = EntropyScorer(
        bundle.arch_spec._inner,
        {qid: loc._inner for qid, loc in bundle.traced_target.items()},
        [loc._inner for loc in bundle.blocked_locations],
    )
    controller = EntropyTreeController(
        reducer=reducer,
        arch_spec=bundle.arch_spec,
        target=bundle.traced_target,
        root_node_id=bundle.root_node_id,
        best_buffer_size=bundle.best_buffer_size,
        scorer=scorer,
        blocked_locations=bundle.blocked_locations,
        qid_label_map=bundle.local_to_global_qid or None,
        blocked_location_labels=bundle.location_to_global_qid or None,
    )
    controller.attach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
