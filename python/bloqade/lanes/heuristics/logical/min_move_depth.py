"""Exhaustive initial layout search for the Gemini Logical architecture."""

from dataclasses import dataclass
from itertools import pairwise
from math import hypot

from kirin import interp

from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic


@dataclass(frozen=True)
class _Route:
    group: int
    depth: int
    distance: float


@dataclass
class LogicalLayoutHeuristicMinMoveDepth(LogicalLayoutHeuristic):
    """Find the home layout with the smallest estimated CZ move depth.

    This opt-in heuristic searches every relevant register-to-home assignment.
    Within each CZ stage it groups same-column moves by column and signed row
    displacement, and cross-column moves by the left atom's signed displacement
    to the right atom's row. Same-column groups cost one move; cross-column
    groups cost two. Ties are broken by the sum of one exact lane-path length
    per group, then by home-address order. Left and right word buses are never
    grouped together. Pinned ``qalloc_at`` locations remain fixed.

    The result optimizes this bus-path estimate, not the compiled move trace.
    A dense ten-qubit circuit can require searching up to 10! placements.

    Use ``LogicalPipeline(layout_heuristic=LogicalLayoutHeuristicMinMoveDepth())``
    to select it; the compiler's default greedy heuristic is unchanged.
    """

    def _geometry(
        self,
    ) -> tuple[tuple[LocationAddress, ...], tuple[tuple[_Route, ...], ...]]:
        homes = tuple(sorted(self.arch_spec.home_sites))
        positions = {home: self.arch_spec.get_position(home) for home in homes}
        xs = sorted({pos[0] for pos in positions.values()})
        ys = sorted({pos[1] for pos in positions.values()})
        if (
            len(xs) != 2
            or len(ys) * 2 != len(homes)
            or len({home.zone_id for home in homes}) != 1
            or len(positions.values()) != len(set(positions.values()))
        ):
            raise ValueError(
                "minimum-move-depth layout requires a two-column, one-zone "
                "logical architecture with one home per column and row"
            )

        grid = {(xs.index(x), ys.index(y)): home for home, (x, y) in positions.items()}
        if len(grid) != len(homes):
            raise ValueError(
                "logical home positions must form a complete two-column grid"
            )
        partners = {home: self.arch_spec.get_cz_partner(home) for home in homes}
        if any(partner is None for partner in partners.values()):
            raise ValueError("every logical home must have a CZ staging partner")

        def distance(src: LocationAddress, dst: LocationAddress) -> float:
            lane = self.arch_spec.get_lane_address(src, dst)
            if lane is None:
                raise ValueError(
                    f"logical architecture has no lane from {src} to {dst}"
                )
            points = self.arch_spec.get_path(lane)
            return sum(
                hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in pairwise(points)
            )

        coordinates = {
            home: (xs.index(position[0]), ys.index(position[1]))
            for home, position in positions.items()
        }
        width = 2 * len(ys) - 1
        routes: list[tuple[_Route, ...]] = []
        for source in homes:
            source_col, source_row = coordinates[source]
            row_routes: list[_Route] = []
            for target in homes:
                if source == target:
                    row_routes.append(_Route(-1, 0, 0.0))
                    continue
                target_col, target_row = coordinates[target]
                if source_col == target_col:
                    row_routes.append(
                        _Route(
                            source_col * width + target_row - source_row + len(ys) - 1,
                            1,
                            distance(source, partners[target]),  # type: ignore[arg-type]
                        )
                    )
                else:
                    left, right = (
                        (source, target) if source_col == 0 else (target, source)
                    )
                    left_row = coordinates[left][1]
                    right_row = coordinates[right][1]
                    left_at_right_row = partners[grid[0, right_row]]
                    right_staging = partners[right]
                    assert left_at_right_row is not None and right_staging is not None
                    row_routes.append(
                        _Route(
                            2 * width + right_row - left_row + len(ys) - 1,
                            2,
                            distance(left, left_at_right_row)
                            + distance(left_at_right_row, right_staging),
                        )
                    )
            routes.append(tuple(row_routes))
        return homes, tuple(routes)

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, LocationAddress] | None = None,
    ) -> tuple[LocationAddress, ...]:
        pinned = {} if pinned is None else pinned
        self._validate_pinned(all_qubits, pinned)
        if len(all_qubits) > self.arch_spec.max_qubits:
            raise interp.InterpreterError(
                f"Number of qubits in circuit ({len(all_qubits)}) exceeds "
                f"maximum supported by logical architecture ({self.arch_spec.max_qubits})"
            )
        homes, routes = self._geometry()
        if len(all_qubits) > len(homes):
            raise ValueError("not enough logical home sites for all qubits")

        qubits = tuple(sorted(all_qubits))
        home_index = {home: index for index, home in enumerate(homes)}
        active = {qubit for stage in stages for pair in stage for qubit in pair}
        if not active <= set(qubits):
            raise ValueError("CZ stage refers to a qubit absent from all_qubits")
        if any(source == target for stage in stages for source, target in stage):
            raise ValueError("CZ stage cannot pair a qubit with itself")

        # Inactive qubits do not influence the score. Fill them in address order
        # after searching the active assignments to obtain the lexicographic tie.
        search_qubits = tuple(q for q in qubits if q in active and q not in pinned)
        layout = {q: home_index[home] for q, home in pinned.items()}
        used = sum(1 << slot for slot in layout.values())
        by_last: dict[int, list[tuple[int, int, int]]] = {q: [] for q in search_qubits}
        initial_pairs: list[tuple[int, int, int]] = []
        for layer, stage in enumerate(stages):
            for source, target in stage:
                pending = [q for q in (source, target) if q not in pinned]
                if pending:
                    by_last[max(pending)].append((layer, source, target))
                else:
                    initial_pairs.append((layer, source, target))

        masks = [0] * len(stages)
        initial_depth = 0
        for layer, source, target in initial_pairs:
            route = routes[layout[source]][layout[target]]
            bit = 1 << route.group
            if not masks[layer] & bit:
                masks[layer] |= bit
                initial_depth += route.depth

        best_depth = float("inf")
        best_distance = float("inf")
        best_slots: tuple[int, ...] | None = None

        def score_distance() -> float:
            total = 0.0
            for stage in stages:
                seen = 0
                for source, target in stage:
                    route = routes[layout[source]][layout[target]]
                    bit = 1 << route.group
                    if not seen & bit:
                        seen |= bit
                        total += route.distance
            return total

        def visit(index: int, occupied: int, depth: int) -> None:
            nonlocal best_depth, best_distance, best_slots
            if depth > best_depth:
                return
            if index == len(search_qubits):
                free = (
                    slot for slot in range(len(homes)) if not occupied & (1 << slot)
                )
                for qubit in qubits:
                    if qubit not in layout:
                        layout[qubit] = next(free)
                slots = tuple(layout[qubit] for qubit in qubits)
                distance = score_distance()
                if (depth, distance, slots) < (
                    best_depth,
                    best_distance,
                    best_slots if best_slots is not None else (),
                ):
                    best_depth, best_distance, best_slots = depth, distance, slots
                for qubit in qubits:
                    if qubit not in pinned and qubit not in active:
                        del layout[qubit]
                return

            qubit = search_qubits[index]
            for slot in range(len(homes)):
                if occupied & (1 << slot):
                    continue
                layout[qubit] = slot
                changes: list[tuple[int, int]] = []
                next_depth = depth
                for layer, source, target in by_last[qubit]:
                    route = routes[layout[source]][layout[target]]
                    bit = 1 << route.group
                    if not masks[layer] & bit:
                        changes.append((layer, masks[layer]))
                        masks[layer] |= bit
                        next_depth += route.depth
                visit(index + 1, occupied | (1 << slot), next_depth)
                for layer, old_mask in reversed(changes):
                    masks[layer] = old_mask
                del layout[qubit]

        visit(0, used, initial_depth)
        assert best_slots is not None
        return tuple(homes[slot] for slot in best_slots)
