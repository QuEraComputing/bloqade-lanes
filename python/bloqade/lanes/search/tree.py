"""Configuration tree for exploring valid atom move programs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from bloqade.lanes.layout import Direction, LaneAddress, LocationAddress, MoveType
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.search.configuration import ConfigurationNode

if TYPE_CHECKING:
    from bloqade.lanes.search.generators import MoveGenerator


class InvalidMoveError(Exception):
    """Raised when a generator produces an invalid move set in strict mode."""


class ExpansionStatus(str, Enum):
    """Outcome status for one move-set expansion attempt."""

    CREATED_CHILD = "created_child"
    ALREADY_CHILD = "already_child"
    TRANSPOSITION_SEEN = "transposition_seen"
    INVALID_LANE = "invalid_lane"
    COLLISION = "collision"


@dataclass(frozen=True)
class ExpansionOutcome:
    """Detailed result for one attempted move-set expansion."""

    move_set: frozenset[LaneAddress]
    status: ExpansionStatus
    child: ConfigurationNode | None = None
    existing_node: ConfigurationNode | None = None
    error_message: str | None = None


@dataclass
class ConfigurationTree:
    """Tree that explores the space of valid atom configurations.

    Starting from an initial placement, the tree tracks created nodes and
    validates move sets. Move generation and node expansion are
    delegated to MoveGenerator implementations.

    NOTE: If deadlock density is high, consider refactoring to a DAG
    (directed acyclic graph) where nodes can have multiple parents.
    This enables backward propagation of deadlock information — when a
    subtree is exhausted, all parents are notified and can prune early.

    Transposition behavior is branch-local by design: only ancestor-chain
    revisits are treated as TRANSPOSITION_SEEN. Equivalent configurations
    reached through different branches are intentionally explored as distinct
    nodes to preserve path diversity.
    """

    arch_spec: ArchSpec
    root: ConfigurationNode
    blocked_locations: frozenset[LocationAddress] = frozenset()
    path_finder: PathFinder = field(init=False, repr=False)
    # Registry of allocated nodes keyed by object identity.
    # This intentionally does NOT deduplicate by configuration key.
    seen: dict[int, ConfigurationNode] = field(
        default_factory=dict, init=False, repr=False
    )
    _lanes_by_triplet: dict[
        tuple[MoveType, int, int, Direction], tuple[LaneAddress, ...]
    ] = field(default_factory=dict, init=False, repr=False)
    _outgoing_lanes_by_src: dict[LocationAddress, tuple[LaneAddress, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.path_finder = PathFinder(self.arch_spec)
        self.seen[id(self.root)] = self.root
        self._build_lane_indexes()

    def _build_lane_indexes(self) -> None:
        """Precomputes all lane mappings once (meant to be called at tree construction time)."""
        lanes_by_triplet: dict[
            tuple[MoveType, int, int, Direction], list[LaneAddress]
        ] = {}
        outgoing: dict[LocationAddress, list[LaneAddress]] = defaultdict(list)

        for zone_id, zone in enumerate(self.arch_spec.zones):
            for mt in (MoveType.SITE, MoveType.WORD):
                buses = zone.site_buses if mt == MoveType.SITE else zone.word_buses
                for bus_id, bus in enumerate(buses):
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        key = (mt, zone_id, bus_id, direction)
                        if key not in lanes_by_triplet:
                            lanes_by_triplet[key] = []
                        lanes_for_key = lanes_by_triplet[key]
                        if mt == MoveType.SITE:
                            for word_id in zone.words_with_site_buses:
                                for site_id in bus.src:
                                    lane = LaneAddress(
                                        mt,
                                        word_id,
                                        site_id,
                                        bus_id,
                                        direction,
                                        zone_id,
                                    )
                                    src, _ = self.arch_spec.get_endpoints(lane)
                                    lanes_for_key.append(lane)
                                    outgoing[src].append(lane)
                        else:
                            for word_id in bus.src:
                                for site_id in zone.sites_with_word_buses:
                                    lane = LaneAddress(
                                        mt,
                                        word_id,
                                        site_id,
                                        bus_id,
                                        direction,
                                        zone_id,
                                    )
                                    src, _ = self.arch_spec.get_endpoints(lane)
                                    lanes_for_key.append(lane)
                                    outgoing[src].append(lane)

        self._lanes_by_triplet = {
            key: tuple(values) for key, values in lanes_by_triplet.items()
        }
        self._outgoing_lanes_by_src = {
            src: tuple(values) for src, values in outgoing.items()
        }

    @classmethod
    def from_initial_placement(
        cls,
        arch_spec: ArchSpec,
        placement: dict[int, LocationAddress],
        blocked_locations: frozenset[LocationAddress] = frozenset(),
    ) -> ConfigurationTree:
        """Create a tree from an initial qubit placement.

        Args:
            arch_spec: Architecture specification for lane validation.
            placement: Mapping of qubit IDs to their initial locations.
            blocked_locations: Locations occupied by atoms outside this placement
                (e.g. other qubits not involved in the current operation).
                These are treated as immovable obstacles during path search.

        Returns:
            A new ConfigurationTree rooted at the given placement.
        """
        root = ConfigurationNode(
            configuration=dict(placement),
            external_occupied=blocked_locations,
        )
        return cls(
            arch_spec=arch_spec,
            root=root,
            blocked_locations=frozenset(blocked_locations),
        )

    def outgoing_lanes(self, source: LocationAddress) -> tuple[LaneAddress, ...]:
        """Return all precomputed outgoing lanes from source."""
        return self._outgoing_lanes_by_src.get(source, ())

    def lanes_for(
        self,
        move_type: MoveType,
        bus_id: int,
        direction: Direction,
        zone_id: int | None = None,
    ) -> tuple[LaneAddress, ...]:
        """Return precomputed lanes for a bus context.

        If zone_id is None, returns lanes across all zones for the given
        (move_type, bus_id, direction). Otherwise scopes to one zone.
        """
        if zone_id is not None:
            return self._lanes_by_triplet.get(
                (move_type, zone_id, bus_id, direction), ()
            )

        lanes: list[LaneAddress] = []
        for zid in range(len(self.arch_spec.zones)):
            lanes.extend(
                self._lanes_by_triplet.get((move_type, zid, bus_id, direction), ())
            )
        return tuple(lanes)

    def lane_for_source(
        self,
        move_type: MoveType,
        bus_id: int,
        direction: Direction,
        source: LocationAddress,
    ) -> LaneAddress | None:
        """Return lane matching source for a bus context, if any."""
        for lane in self.lanes_for(
            move_type=move_type,
            bus_id=bus_id,
            direction=direction,
            zone_id=source.zone_id,
        ):
            src, _ = self.arch_spec.get_endpoints(lane)
            if src == source:
                return lane
        return None

    def _ancestor_with_config_key(
        self, node: ConfigurationNode, target_key: object
    ) -> ConfigurationNode | None:
        """Return matching ancestor on this branch for a configuration key."""
        ancestor: ConfigurationNode | None = node
        while ancestor is not None:
            if ancestor.config_key == target_key:
                return ancestor
            ancestor = ancestor.parent
        return None

    def valid_lanes(
        self,
        node: ConfigurationNode,
        move_type: MoveType | None = None,
        bus_id: int | None = None,
        direction: Direction | None = None,
    ) -> Iterator[LaneAddress]:
        """Yield valid individual lane addresses from a configuration.

        A lane is valid if its source is occupied and its destination
        is not occupied. Optionally filter by move_type, bus_id, and
        direction — None means include all.

        Args:
            node: The configuration to query.
            move_type: Filter to this move type, or None for all.
            bus_id: Filter to this bus ID, or None for all.
            direction: Filter to this direction, or None for all.

        Yields:
            Valid LaneAddress values.
        """
        occupied = node.occupied_locations
        blocked = self.blocked_locations

        # Iterate all registered (move_type, zone_id, bus_id, direction) keys,
        # filtering by the caller's constraints.
        for (mt, zid, bid, d), lanes in self._lanes_by_triplet.items():
            if move_type is not None and mt != move_type:
                continue
            if bus_id is not None and bid != bus_id:
                continue
            if direction is not None and d != direction:
                continue
            for lane in lanes:
                src, dst = self.arch_spec.get_endpoints(lane)
                if src in occupied and dst not in occupied and dst not in blocked:
                    yield lane

    def apply_move_set(
        self,
        node: ConfigurationNode,
        move_set: frozenset[LaneAddress],
        strict: bool = True,
    ) -> ConfigurationNode | None:
        """Apply a move set to a node, returning a new child or None.

        Resolves lane endpoints, checks for collisions, and creates
        a child node if valid.

        The move set is first validated against the arch spec (AOD
        geometry, consistency, bus membership).  This check runs as an
        ``assert`` and is independent of *strict* — it signals an
        internal bug, not a recoverable runtime condition.  It can be
        disabled with ``python -O``.

        Args:
            node: The node to apply moves to.
            move_set: The set of lane addresses to apply.
            strict: If True (default), raises InvalidMoveError when a
                move set causes a collision. If False, silently returns
                None for invalid moves.

        Returns:
            A new ConfigurationNode, or None if the move is invalid and
            strict=False.

        Raises:
            AssertionError: If the move set fails lane-group validation
                (regardless of *strict*; disabled with ``python -O``).
            InvalidMoveError: If strict=True and the move set causes a
                collision or contains an invalid lane address.
        """
        outcome = self.try_move_set(node, move_set, strict=strict)
        return (
            outcome.child if outcome.status == ExpansionStatus.CREATED_CHILD else None
        )

    def try_move_set(
        self,
        node: ConfigurationNode,
        move_set: frozenset[LaneAddress],
        strict: bool = True,
    ) -> ExpansionOutcome:
        """Attempt one move set and return a detailed outcome.

        In strict mode, invalid lanes and collisions raise InvalidMoveError
        (matching apply_move_set behavior). In non-strict mode, they are
        returned as INVALID_LANE/COLLISION outcomes.
        """
        lane_errors = self.arch_spec.check_lane_group(list(move_set))
        if lane_errors:
            msg = f"Move set failed lane-group validation: {'; '.join(str(e) for e in lane_errors)}"
            if strict:
                raise InvalidMoveError(msg)
            return ExpansionOutcome(
                move_set=move_set,
                status=ExpansionStatus.INVALID_LANE,
                error_message=msg,
            )

        existing_child = node.children.get(move_set)
        if existing_child is not None:
            return ExpansionOutcome(
                move_set=move_set,
                status=ExpansionStatus.ALREADY_CHILD,
                child=existing_child,
            )

        new_config = dict(node.configuration)
        occupied = node.occupied_locations
        blocked = self.blocked_locations

        for lane in move_set:
            try:
                src, dst = self.arch_spec.get_endpoints(lane)
            except Exception as e:
                msg = f"Invalid lane address {lane!r}: {e}"
                if strict:
                    raise InvalidMoveError(msg) from e
                return ExpansionOutcome(
                    move_set=move_set,
                    status=ExpansionStatus.INVALID_LANE,
                    error_message=msg,
                )

            qid = node.get_qubit_at(src)
            if qid is None:
                continue

            # Bus src and dst are disjoint sets, so within a single-bus
            # move set, two sources cannot map to the same destination.
            # We only need to check against stationary atoms.
            if dst in occupied or dst in blocked:
                blocker = node.get_qubit_at(dst)
                blocker_text = (
                    f"qubit {blocker}" if blocker is not None else "a blocked location"
                )
                msg = (
                    f"Collision: qubit {qid} moving to {dst!r} "
                    f"which is occupied by {blocker_text}"
                )
                if strict:
                    raise InvalidMoveError(msg)
                return ExpansionOutcome(
                    move_set=move_set,
                    status=ExpansionStatus.COLLISION,
                    error_message=msg,
                )

            new_config[qid] = dst

        new_node = ConfigurationNode(
            configuration=new_config,
            parent=node,
            parent_moves=move_set,
            depth=node.depth + 1,
            external_occupied=node.external_occupied,
        )
        matching_ancestor = self._ancestor_with_config_key(node, new_node.config_key)
        if matching_ancestor is not None:
            return ExpansionOutcome(
                move_set=move_set,
                status=ExpansionStatus.TRANSPOSITION_SEEN,
                existing_node=matching_ancestor,
            )

        # Register by object identity so equivalent configurations reached
        # through different histories are still explored independently.
        self.seen[id(new_node)] = new_node
        node.children[move_set] = new_node
        return ExpansionOutcome(
            move_set=move_set,
            status=ExpansionStatus.CREATED_CHILD,
            child=new_node,
        )

    def expand_node_detailed(
        self,
        node: ConfigurationNode,
        generator: MoveGenerator,
        strict: bool = True,
    ) -> Iterator[ExpansionOutcome]:
        """Expand a node and yield one outcome per generated candidate."""
        for move_set in generator.generate(node, self):
            yield self.try_move_set(node, move_set, strict=strict)

    def expand_node(
        self,
        node: ConfigurationNode,
        generator: MoveGenerator,
        strict: bool = True,
    ) -> list[ConfigurationNode]:
        """Expand a node using the given generator.

        Generates candidate move sets, validates each (collision checks),
        and creates child nodes.

        Args:
            node: The node to expand.
            generator: Produces candidate move sets.
            strict: If True, raises on invalid moves. If False, skips them.

        Returns:
            List of newly created child nodes (may be empty).
        """
        children: list[ConfigurationNode] = []
        for outcome in self.expand_node_detailed(node, generator, strict=strict):
            if outcome.status == ExpansionStatus.CREATED_CHILD:
                assert outcome.child is not None
                children.append(outcome.child)
        return children
