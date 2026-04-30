"""Type stubs for the _native PyO3 extension module."""

from typing import Optional, final

from bloqade.lanes.bytecode.exceptions import (
    LaneGroupError,
    LocationGroupError,
)

# ── Enums ──

@final
class Direction:
    """Atom movement direction along a bus.

    Attributes:
        FORWARD: Movement from source to destination (value 0).
        BACKWARD: Movement from destination to source (value 1).
    """

    FORWARD: Direction
    BACKWARD: Direction
    @property
    def name(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

@final
class MoveType:
    """Type of bus used for an atom move operation.

    Attributes:
        SITE: Moves atoms between sites within a word (value 0).
        WORD: Moves atoms between words (value 1).
        ZONE: Moves atoms between zones (value 2).
    """

    SITE: MoveType
    WORD: MoveType
    ZONE: MoveType
    @property
    def name(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

# ── Address Types ──

@final
class LocationAddress:
    """Bit-packed atom location address (zone + word + site).

    Encodes ``zone_id`` (8 bits), ``word_id`` (16 bits), and
    ``site_id`` (16 bits) into a 64-bit word.

    Layout: ``[zone_id:8][word_id:16][site_id:16][pad:24]``

    Args:
        zone_id (int): Zone identifier (0..255).
        word_id (int): Word identifier (0..65535).
        site_id (int): Site identifier within the word (0..65535).
    """

    def __init__(self, zone_id: int, word_id: int, site_id: int) -> None: ...
    @property
    def zone_id(self) -> int:
        """Zone identifier."""
        ...

    @property
    def word_id(self) -> int:
        """Word identifier."""
        ...

    @property
    def site_id(self) -> int:
        """Site identifier within the word."""
        ...

    def encode(self) -> int:
        """Encode to a 64-bit packed integer.

        Returns:
            int: The 64-bit packed representation.
        """
        ...

    @staticmethod
    def decode(bits: int) -> LocationAddress:
        """Decode a 64-bit packed integer into a LocationAddress.

        Args:
            bits (int): The 64-bit packed representation.

        Returns:
            LocationAddress: The decoded address.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class LaneAddress:
    """Bit-packed lane address for atom move operations.

    Encodes direction (1 bit), move_type (2 bits), zone_id (8 bits),
    word_id (16 bits), site_id (16 bits), and bus_id (16 bits) across
    two 32-bit data words, returned as a combined 64-bit value.

    Layout:
        data0: ``[word_id:16][site_id:16]``
        data1: ``[dir:1][mt:2][zone_id:8][pad:5][bus_id:16]``

    Args:
        move_type (MoveType): SITE, WORD, or ZONE.
        zone_id (int): Zone identifier (0..255).
        word_id (int): Word identifier (0..65535).
        site_id (int): Site identifier within the word (0..65535).
        bus_id (int): Bus identifier (0..65535).
        direction (Direction): Forward or Backward. Default: Direction.FORWARD.
    """

    def __init__(
        self,
        move_type: MoveType,
        zone_id: int,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = ...,
    ) -> None: ...
    @property
    def direction(self) -> Direction:
        """Movement direction (FORWARD or BACKWARD)."""
        ...

    @property
    def move_type(self) -> MoveType:
        """Bus type (SITE, WORD, or ZONE)."""
        ...

    @property
    def zone_id(self) -> int:
        """Zone identifier."""
        ...

    @property
    def word_id(self) -> int:
        """Word identifier."""
        ...

    @property
    def site_id(self) -> int:
        """Site identifier within the word."""
        ...

    @property
    def bus_id(self) -> int:
        """Bus identifier."""
        ...

    def encode(self) -> int:
        """Encode to a 64-bit packed integer.

        Returns:
            int: The 64-bit packed representation (data0 | data1 << 32).
        """
        ...

    @staticmethod
    def decode(bits: int) -> LaneAddress:
        """Decode a 64-bit packed integer into a LaneAddress.

        Args:
            bits (int): The 64-bit packed representation.

        Returns:
            LaneAddress: The decoded address.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ZoneAddress:
    """Bit-packed zone address.

    Encodes a zone identifier (8 bits) into a 32-bit value.

    Layout: ``[pad:24][zone_id:8]``

    Args:
        zone_id (int): Zone identifier (0..255).
    """

    def __init__(self, zone_id: int) -> None: ...
    @property
    def zone_id(self) -> int:
        """Zone identifier."""
        ...

    def encode(self) -> int:
        """Encode to a 32-bit packed integer.

        Returns:
            int: The 32-bit packed representation.
        """
        ...

    @staticmethod
    def decode(bits: int) -> ZoneAddress:
        """Decode a 32-bit packed integer into a ZoneAddress.

        Args:
            bits (int): The 32-bit packed representation.

        Returns:
            ZoneAddress: The decoded address.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

# ── Arch Spec Types ──

@final
class Grid:
    """Coordinate grid defining physical positions for atom sites.

    A grid defines positions via a start coordinate and spacing values.
    The x-coordinates are ``[x_start, x_start + x_spacing[0], ...]``
    (cumulative sum of spacings from the start). Same for y.

    Args:
        x_start (float): X-coordinate of the first grid point.
        y_start (float): Y-coordinate of the first grid point.
        x_spacing (list[float]): Spacing between consecutive x-coordinates.
        y_spacing (list[float]): Spacing between consecutive y-coordinates.
    """

    def __init__(
        self,
        x_start: float,
        y_start: float,
        x_spacing: list[float],
        y_spacing: list[float],
    ) -> None: ...
    @classmethod
    def from_positions(cls, x_positions: list[float], y_positions: list[float]) -> Grid:
        """Construct a Grid from explicit position arrays.

        The first element becomes the start value and consecutive differences
        become the spacing vector.

        Args:
            x_positions (list[float]): X-coordinates (at least one element).
            y_positions (list[float]): Y-coordinates (at least one element).

        Returns:
            Grid: The constructed grid.

        Raises:
            ValueError: If either list is empty.
        """
        ...

    @property
    def num_x(self) -> int:
        """Number of x-axis grid points (``len(x_spacing) + 1``)."""
        ...

    @property
    def num_y(self) -> int:
        """Number of y-axis grid points (``len(y_spacing) + 1``)."""
        ...

    @property
    def x_start(self) -> float:
        """X-coordinate of the first grid point."""
        ...

    @property
    def y_start(self) -> float:
        """Y-coordinate of the first grid point."""
        ...

    @property
    def x_spacing(self) -> list[float]:
        """Spacing between consecutive x-coordinates."""
        ...

    @property
    def y_spacing(self) -> list[float]:
        """Spacing between consecutive y-coordinates."""
        ...

    @property
    def x_positions(self) -> list[float]:
        """Computed x-axis coordinate values."""
        ...

    @property
    def y_positions(self) -> list[float]:
        """Computed y-axis coordinate values."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Word:
    """A group of atom sites that share a coordinate grid.

    Each word contains a fixed number of sites. Sites are positioned on the
    parent zone's grid via ``[x_idx, y_idx]`` index pairs.

    Args:
        sites (list[tuple[int, int]]): Site positions as ``(x_idx, y_idx)`` grid index pairs.

    Note: A word's identity is determined by its position in the ``ArchSpec.words`` list.
    """

    def __init__(self, sites: list[tuple[int, int]]) -> None: ...
    @property
    def sites(self) -> list[tuple[int, int]]:
        """Site positions as ``(x_idx, y_idx)`` grid index pairs."""
        ...

    def __repr__(self) -> str: ...

@final
class SiteBus:
    """A transport bus that maps source sites to destination sites within a zone.

    The ``src`` and ``dst`` lists are parallel arrays of site indices:
    ``src[i]`` maps to ``dst[i]``.

    Args:
        src (list[int]): Source site indices.
        dst (list[int]): Destination site indices.
    """

    def __init__(self, src: list[int], dst: list[int]) -> None: ...
    @property
    def src(self) -> list[int]:
        """Source site indices."""
        ...

    @property
    def dst(self) -> list[int]:
        """Destination site indices."""
        ...

    def resolve_forward(self, src: int) -> Optional[int]:
        """Map a source site to its destination.

        Args:
            src (int): Source site index.

        Returns:
            int: The destination site index, or None if not found.
        """
        ...

    def resolve_backward(self, dst: int) -> Optional[int]:
        """Map a destination site back to its source.

        Args:
            dst (int): Destination site index.

        Returns:
            int: The source site index, or None if not found.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class WordBus:
    """A transport bus that maps source words to destination words within a zone.

    The ``src`` and ``dst`` lists are parallel arrays of word indices:
    ``src[i]`` maps to ``dst[i]``.

    Args:
        src (list[int]): Source word indices.
        dst (list[int]): Destination word indices.
    """

    def __init__(self, src: list[int], dst: list[int]) -> None: ...
    @property
    def src(self) -> list[int]:
        """Source word indices."""
        ...

    @property
    def dst(self) -> list[int]:
        """Destination word indices."""
        ...

    def resolve_forward(self, src: int) -> Optional[int]:
        """Map a source word to its destination.

        Args:
            src (int): Source word index.

        Returns:
            int: The destination word index, or None if not found.
        """
        ...

    def resolve_backward(self, dst: int) -> Optional[int]:
        """Map a destination word back to its source.

        Args:
            dst (int): Destination word index.

        Returns:
            int: The source word index, or None if not found.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ZoneBus:
    """An inter-zone transport bus that maps source zone+word pairs to destinations.

    The ``src`` and ``dst`` lists are parallel arrays of ``(zone_id, word_id)``
    tuples: ``src[i]`` maps to ``dst[i]``.

    Args:
        src (list[tuple[int, int]]): Source ``(zone_id, word_id)`` pairs.
        dst (list[tuple[int, int]]): Destination ``(zone_id, word_id)`` pairs.
    """

    def __init__(
        self,
        src: list[tuple[int, int]],
        dst: list[tuple[int, int]],
    ) -> None: ...
    @property
    def src(self) -> list[tuple[int, int]]:
        """Source ``(zone_id, word_id)`` pairs."""
        ...

    @property
    def dst(self) -> list[tuple[int, int]]:
        """Destination ``(zone_id, word_id)`` pairs."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Zone:
    """A logical zone grouping words with a shared coordinate grid and buses.

    Each zone owns its grid and the site/word buses that operate within it.

    Args:
        grid (Grid): Coordinate grid for all words in this zone.
        site_buses (list[SiteBus]): Site buses within this zone.
        word_buses (list[WordBus]): Word buses within this zone.
        words_with_site_buses (list[int]): Word IDs with site-bus transport.
        sites_with_word_buses (list[int]): Site indices with word-bus transport.
        entangling_pairs (Optional[list[tuple[int, int]]]): Pairs of word IDs
            within this zone for CZ entangling gates. Default = None (no pairs).
    """

    def __init__(
        self,
        name: str,
        grid: Grid,
        site_buses: list[SiteBus],
        word_buses: list[WordBus],
        words_with_site_buses: list[int],
        sites_with_word_buses: list[int],
        entangling_pairs: Optional[list[tuple[int, int]]] = None,
    ) -> None: ...
    @property
    def name(self) -> str:
        """Human-readable zone name."""
        ...

    @property
    def grid(self) -> Grid:
        """Coordinate grid for this zone."""
        ...

    @property
    def site_buses(self) -> list[SiteBus]:
        """Site buses within this zone."""
        ...

    @property
    def word_buses(self) -> list[WordBus]:
        """Word buses within this zone."""
        ...

    @property
    def words_with_site_buses(self) -> list[int]:
        """Word IDs with site-bus transport capability."""
        ...

    @property
    def sites_with_word_buses(self) -> list[int]:
        """Site indices that participate in word-bus transport."""
        ...

    @property
    def entangling_pairs(self) -> list[tuple[int, int]]:
        """Pairs of word IDs within this zone for CZ entangling gates."""
        ...

    def __repr__(self) -> str: ...

@final
class Mode:
    """A named operational mode for the device.

    Modes define subsets of zones and the bitstring ordering used for
    measurement results.

    Args:
        name (str): Human-readable mode name.
        zones (list[int]): Zone IDs active in this mode.
        bitstring_order (list[LocationAddress]): Bit-to-location mapping.
    """

    def __init__(
        self,
        name: str,
        zones: list[int],
        bitstring_order: list[LocationAddress],
    ) -> None: ...
    @property
    def name(self) -> str:
        """Human-readable mode name."""
        ...

    @property
    def zones(self) -> list[int]:
        """Zone IDs active in this mode."""
        ...

    @property
    def bitstring_order(self) -> list[LocationAddress]:
        """Bit-to-location mapping for measurement results."""
        ...

    def __repr__(self) -> str: ...

@final
class TransportPath:
    """A transport path for a lane, defined by waypoints.

    The lane is identified by a ``LaneAddress`` which encodes the direction,
    move type, zone, word, site, and bus.

    Args:
        lane (LaneAddress): Lane address identifying the transport lane.
        waypoints (list[tuple[float, float]]): Sequence of ``(x, y)`` coordinate waypoints.

    Note: In JSON, the lane is serialized as a 16-digit hex string (e.g. ``"0xC000000000000000"``).
    """

    def __init__(
        self,
        lane: LaneAddress,
        waypoints: list[tuple[float, float]],
    ) -> None: ...
    @property
    def lane(self) -> LaneAddress:
        """Decoded lane address."""
        ...

    @property
    def lane_encoded(self) -> int:
        """Raw encoded lane address as a 64-bit integer."""
        ...

    @property
    def waypoints(self) -> list[tuple[float, float]]:
        """Sequence of ``(x, y)`` coordinate waypoints."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ArchSpec:
    """Architecture specification for a quantum device.

    Describes the full hardware topology: words, zones (each owning a grid,
    intra-zone buses, and entangling pairs), inter-zone buses, operational
    modes, and device capabilities.

    Args:
        version (tuple[int, int]): Spec version as ``(major, minor)``.
        words (list[Word]): Word definitions.
        zones (list[Zone]): Zone definitions (each owns a grid, buses, and entangling pairs).
        zone_buses (list[ZoneBus]): Inter-zone word buses.
        modes (list[Mode]): Operational modes.
        paths (Optional[list[TransportPath]]): AOD transport paths, default = None.
        feed_forward (bool): Whether the device supports mid-circuit measurement. Default = False.
        atom_reloading (bool): Whether the device supports atom reloading. Default = False.
        blockade_radius (Optional[float]): Rydberg blockade radius (µm). When set, this
            indicates the radius associated with the architecture and is typically
            used to interpret entangling pairs. It is metadata; this constructor
            does not itself verify that the pairs match the radius. Default = None.
    """

    def __init__(
        self,
        version: tuple[int, int],
        words: list[Word],
        zones: list[Zone],
        zone_buses: list[ZoneBus],
        modes: list[Mode],
        paths: Optional[list[TransportPath]] = None,
        feed_forward: bool = False,
        atom_reloading: bool = False,
        blockade_radius: Optional[float] = None,
    ) -> None: ...
    @staticmethod
    def from_json(json: str) -> ArchSpec:
        """Parse an architecture spec from a JSON string.

        Args:
            json (str): JSON string containing the architecture spec.

        Returns:
            ArchSpec: The parsed architecture spec.

        Raises:
            ValueError: If the JSON is malformed or missing required fields.
        """
        ...

    @staticmethod
    def from_json_validated(json: str) -> ArchSpec:
        """Parse an architecture spec from JSON and validate it.

        Equivalent to calling ``from_json()`` followed by ``validate()``.

        Args:
            json (str): JSON string containing the architecture spec.

        Returns:
            ArchSpec: The parsed and validated architecture spec.

        Raises:
            ValueError: If the JSON is malformed or missing required fields.
            ArchSpecError: If structural validation fails.
        """
        ...

    def validate(self) -> None:
        """Validate the architecture specification.

        Checks structural constraints (zone coverage, bus consistency,
        site/word bounds, etc.). Collects all errors and raises once
        with the full list.

        Raises:
            ArchSpecError: With ``.errors`` list containing individual
                error subclass instances.
        """
        ...

    @property
    def version(self) -> tuple[int, int]:
        """Spec version as ``(major, minor)``."""
        ...

    @property
    def words(self) -> list[Word]:
        """Word definitions."""
        ...

    @property
    def zones(self) -> list[Zone]:
        """Zone definitions."""
        ...

    @property
    def zone_buses(self) -> list[ZoneBus]:
        """Inter-zone word buses."""
        ...

    @property
    def modes(self) -> list[Mode]:
        """Operational modes."""
        ...

    @property
    def sites_per_word(self) -> int:
        """Number of sites in each word (0 if no words)."""
        ...

    @property
    def feed_forward(self) -> bool:
        """Whether the device supports mid-circuit measurement with classical feedback."""
        ...

    @property
    def atom_reloading(self) -> bool:
        """Whether the device supports reloading atoms after initial fill."""
        ...

    @property
    def blockade_radius(self) -> Optional[float]:
        """Rydberg blockade radius (µm), or None if not provided."""
        ...

    @property
    def paths(self) -> Optional[list[TransportPath]]:
        """Transport paths between locations, or None."""
        ...

    def word_by_id(self, id: int) -> Optional[Word]:
        """Look up a word by its index.

        Args:
            id (int): Word index in ``words``.

        Returns:
            Word: The word, or None if not found.
        """
        ...

    def zone_by_id(self, id: int) -> Optional[Zone]:
        """Look up a zone by its index.

        Args:
            id (int): Zone index in ``zones``.

        Returns:
            Zone: The zone, or None if not found.
        """
        ...

    def location_position(self, loc: LocationAddress) -> Optional[tuple[float, float]]:
        """Get the ``(x, y)`` physical position for an atom location.

        Args:
            loc (LocationAddress): The location address to look up.

        Returns:
            tuple[float, float]: The ``(x, y)`` position, or None if the zone,
                word, or site is not found.
        """
        ...

    def lane_endpoints(
        self, lane: LaneAddress
    ) -> Optional[tuple[LocationAddress, LocationAddress]]:
        """Resolve a lane address to its source and destination locations.

        Traces through the appropriate bus (site bus, word bus, or zone bus)
        in the specified direction (forward or backward) to determine which
        two ``LocationAddress`` endpoints the lane connects.

        Args:
            lane (LaneAddress): The lane address to resolve.

        Returns:
            tuple[LocationAddress, LocationAddress]: A ``(src, dst)`` pair, or None if the
                lane references an invalid bus, word, or site.
        """
        ...

    def get_cz_partner(self, loc: LocationAddress) -> Optional[LocationAddress]:
        """Get the CZ partner for a given location.

        Searches the zone's ``entangling_pairs`` for a pair containing the
        location's word_id and returns the partner word in the same zone.

        Args:
            loc (LocationAddress): The location address to look up.

        Returns:
            LocationAddress: The partner location, or None if the word is not
                in any entangling pair within its zone.
        """
        ...
    # -- Derived topology queries (#464 phase 2) --

    def word_partner_map(self) -> dict[int, int]:
        """Bidirectional word partner map from entangling pairs.

        Returns:
            dict[int, int]: word_id → partner_word_id for every word
                appearing in any zone's ``entangling_pairs``.
        """
        ...

    def word_zone_map(self) -> dict[int, int]:
        """Map each word_id to the zone_id that owns it.

        Derived from each zone's ``entangling_pairs``, ``word_buses``,
        and ``words_with_site_buses``. Words not referenced by any zone
        default to zone 0.

        Returns:
            dict[int, int]: word_id → zone_id.
        """
        ...

    def left_cz_word_ids(self) -> list[int]:
        """Sorted left-CZ word IDs.

        The lower word of each entangling pair, plus any word not
        appearing in any pair.

        Returns:
            list[int]: Sorted word IDs.
        """
        ...

    def lane_for_endpoints(
        self, src: LocationAddress, dst: LocationAddress
    ) -> Optional[LaneAddress]:
        """Reverse-lookup: find the lane connecting ``src`` to ``dst``.

        Searches SiteBus, WordBus, and ZoneBus lanes. Exploits the
        LaneAddr encoding to narrow the search to
        ``O(site_buses + word_buses + zone_buses)`` per direction.

        Args:
            src (LocationAddress): Source location.
            dst (LocationAddress): Destination location.

        Returns:
            LaneAddress: The lane, or None if no lane connects them.
        """
        ...

    def zone_location_index(self, loc: LocationAddress, zone_id: int) -> Optional[int]:
        """O(1) flat index of a location within a zone.

        Returns ``word_id * sites_per_word + site_id`` if the location's
        zone matches ``zone_id`` and the word/site are in range, else None.
        """
        ...

    def check_zone(self, addr: ZoneAddress) -> Optional[str]:
        """Check whether a zone address is valid.

        Args:
            addr (ZoneAddress): The zone address to check.

        Returns:
            str: An error message if invalid, or None if valid.
        """
        ...

    def check_locations(
        self, locations: list[LocationAddress]
    ) -> list[LocationGroupError]:
        """Validate a group of location addresses against this architecture.

        Checks for duplicate addresses and invalid zone/word/site combinations.

        Args:
            locations (list[LocationAddress]): Location addresses to validate.

        Returns:
            list[LocationGroupError]: Error instances (empty if all valid).
        """
        ...

    def check_lanes(self, lanes: list[LaneAddress]) -> list[LaneGroupError]:
        """Validate a group of lane addresses against this architecture.

        Checks for duplicates, invalid addresses, bus consistency, and
        AOD constraints.

        Args:
            lanes (list[LaneAddress]): Lane addresses to validate.

        Returns:
            list[LaneGroupError]: Error instances (empty if all valid).
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

# ── Move Solver ──

@final
class SearchStrategy:
    """Search strategy for the move solver."""

    ASTAR: SearchStrategy
    DFS: SearchStrategy
    BFS: SearchStrategy
    GREEDY: SearchStrategy
    IDS: SearchStrategy
    CASCADE_IDS: SearchStrategy
    CASCADE_DFS: SearchStrategy
    CASCADE_ENTROPY: SearchStrategy
    ENTROPY: SearchStrategy

    @property
    def name(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

@final
class DeadlockPolicy:
    """Deadlock handling policy for the move solver."""

    SKIP: DeadlockPolicy
    MOVE_BLOCKERS: DeadlockPolicy
    ALL_MOVES: DeadlockPolicy

    @property
    def name(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

@final
class SolveOptions:
    """Search-tuning parameters for MoveSolver."""

    def __init__(
        self,
        strategy: SearchStrategy = SearchStrategy.ASTAR,
        max_movesets_per_group: int = 3,
        max_goal_candidates: int = 3,
        weight: float = 1.0,
        restarts: int = 1,
        lookahead: bool = False,
        deadlock_policy: DeadlockPolicy = DeadlockPolicy.SKIP,
        w_t: float = 0.05,
        collect_entropy_trace: bool = False,
    ) -> None: ...
    @property
    def strategy(self) -> SearchStrategy: ...
    @property
    def max_movesets_per_group(self) -> int: ...
    @property
    def max_goal_candidates(self) -> int: ...
    @property
    def weight(self) -> float: ...
    @property
    def restarts(self) -> int: ...
    @property
    def lookahead(self) -> bool: ...
    @property
    def deadlock_policy(self) -> DeadlockPolicy: ...
    @property
    def w_t(self) -> float: ...
    @property
    def collect_entropy_trace(self) -> bool: ...
    def __repr__(self) -> str: ...

@final
class SolveResult:
    """Result of a move synthesis solve.

    Always returned by ``MoveSolver.solve()``. Check ``status`` to determine
    whether a solution was found.

    When produced via the ``policy_path`` kwarg, the ``policy_file``,
    ``policy_params``, and ``policy_status`` fields are populated; for the
    strategy-based path they are all ``None``.
    """

    @property
    def status(self) -> str:
        """Status: ``"solved"``, ``"unsolvable"``, or ``"budget_exceeded"``.

        For DSL-path results, also check ``policy_status`` for the full
        terminal state string from the kernel.
        """
        ...

    @property
    def move_layers(self) -> list[list[tuple[int, int, int, int, int, int]]]:
        """Move layers as lists of (direction, move_type, zone_id, word_id, site_id, bus_id) tuples.

        Empty when ``status`` is not ``"solved"``.
        """
        ...

    @property
    def goal_config(self) -> dict[int, LocationAddress]:
        """Goal configuration as qubit_id -> LocationAddress mapping.

        Equals the initial configuration when ``status`` is not ``"solved"``.
        """
        ...

    @property
    def nodes_expanded(self) -> int:
        """Number of nodes expanded during search."""
        ...

    @property
    def cost(self) -> float:
        """Total path cost. 0.0 when ``status`` is not ``"solved"``."""
        ...

    @property
    def deadlocks(self) -> int:
        """Number of deadlocks encountered during search."""
        ...

    @property
    def entropy_trace(self) -> Optional[EntropyTrace]:
        """Optional entropy-search trace when ``collect_entropy_trace=True`` was set."""
        ...

    @property
    def policy_file(self) -> str | None:
        """Echo of the ``.star`` policy file path, or ``None`` if not a DSL solve."""
        ...

    @property
    def policy_params(self) -> str | None:
        """JSON-encoded echo of ``policy_params`` dict, or ``None`` if not a DSL solve.

        Use ``json.loads(result.policy_params)`` to recover the original dict.
        """
        ...

    @property
    def policy_status(self) -> str | None:
        """String representation of the DSL terminal status, or ``None`` if not a
        DSL solve.

        Possible values: ``"solved"``, ``"unsolvable"``, ``"budget_exhausted"``,
        ``"timeout"``, ``"fallback: <detail>"``, ``"syntax_error: <detail>"``,
        ``"runtime_error: <detail>"``, ``"schema_error: <field>"``,
        ``"bad_policy: <detail>"``, ``"starlark_budget"``, ``"starlark_oom"``.
        """
        ...

    def __repr__(self) -> str: ...

@final
class EntropyTrace:
    """Entropy-search trace returned by ``SolveResult.entropy_trace``."""

    @property
    def root_node_id(self) -> int: ...
    @property
    def best_buffer_size(self) -> int: ...
    @property
    def steps(self) -> list[EntropyTraceStep]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class EntropyTraceStep:
    """One step in an entropy-search trace."""

    @property
    def event(self) -> str: ...
    @property
    def node_id(self) -> int: ...
    @property
    def parent_node_id(self) -> Optional[int]: ...
    @property
    def depth(self) -> int: ...
    @property
    def entropy(self) -> int: ...
    @property
    def unresolved_count(self) -> int: ...
    @property
    def moveset(self) -> Optional[list[tuple[int, int, int, int, int, int]]]: ...
    @property
    def candidate_movesets(
        self,
    ) -> list[list[tuple[int, int, int, int, int, int]]]: ...
    @property
    def candidate_index(self) -> Optional[int]: ...
    @property
    def reason(self) -> Optional[str]: ...
    @property
    def state_seen_node_id(self) -> Optional[int]: ...
    @property
    def no_valid_moves_qubit(self) -> Optional[int]: ...
    @property
    def trigger_node_id(self) -> Optional[int]: ...
    @property
    def configuration(self) -> list[tuple[int, int, int, int]]: ...
    @property
    def parent_configuration(self) -> Optional[list[tuple[int, int, int, int]]]: ...
    @property
    def moveset_score(self) -> Optional[float]: ...
    @property
    def best_buffer_node_ids(self) -> list[int]: ...
    def __repr__(self) -> str: ...

@final
class MoveSolver:
    """Reusable move synthesis solver.

    Constructed once from an architecture specification. The constructor
    parses the spec and precomputes lane indexes. Then ``solve()`` can be
    called multiple times with different placements.
    """

    def __init__(self, arch_spec_json: str) -> None: ...
    @staticmethod
    def from_arch_spec(arch: ArchSpec) -> MoveSolver:
        """Create a solver from a native ArchSpec object."""
        ...

    def solve(
        self,
        initial: dict[int, LocationAddress],
        target: dict[int, LocationAddress],
        blocked: list[LocationAddress],
        *,
        max_expansions: int | None = None,
        options: SolveOptions | None = None,
        policy_path: str | None = None,
        policy_params: dict[str, object] | None = None,
        timeout_s: float | None = None,
    ) -> SolveResult:
        """Solve a move synthesis problem.

        Args:
            initial: Mapping of qubit_id to LocationAddress for starting positions.
            target: Mapping of qubit_id to LocationAddress for desired positions.
            blocked: List of LocationAddress for immovable obstacle locations.
            max_expansions: Optional limit on node expansions.
            options: Search-tuning parameters. Defaults to SolveOptions().
            policy_path: Path to a ``.star`` Move Policy DSL file. When
                supplied, routes through ``solve_with_policy`` instead of the
                strategy-based search path.
            policy_params: Free-form dict echoed back in
                ``SolveResult.policy_params`` (JSON-encoded). Only used when
                ``policy_path`` is supplied.
            timeout_s: Wall-clock time limit in seconds for the DSL kernel.
                Only used when ``policy_path`` is supplied.

        Returns:
            SolveResult with status indicating outcome. When ``policy_path``
            is used, check ``policy_status``, ``policy_file``, and
            ``policy_params`` on the result for DSL-specific information.
        """
        ...

    def solve_with_generator(
        self,
        initial: dict[int, LocationAddress],
        blocked: list[LocationAddress],
        controls: list[int],
        targets: list[int],
        generator: DefaultTargetGenerator | None = None,
        max_expansions: Optional[int] = None,
        options: SolveOptions | None = None,
    ) -> MultiSolveResult:
        """Solve using a target generator with shared expansion budget.

        Args:
            initial: Mapping of qubit_id to LocationAddress for starting positions.
            blocked: List of LocationAddress for immovable obstacle locations.
            controls: Control qubit IDs for the CZ gate layer.
            targets: Target qubit IDs for the CZ gate layer.
            generator: Rust-side target generator (currently must be None).
            max_expansions: Total expansion budget across all candidates.
            options: Search-tuning parameters. Defaults to SolveOptions().

        Returns:
            MultiSolveResult with per-candidate debug info.
        """
        ...

    def generate_candidates(
        self,
        initial: dict[int, LocationAddress],
        controls: list[int],
        targets: list[int],
        generator: DefaultTargetGenerator | None = None,
    ) -> list[dict[int, LocationAddress]]:
        """Generate and validate candidate targets without solving.

        Returns only validated candidates as qubit_id -> LocationAddress mappings.
        """
        ...

    def __repr__(self) -> str: ...

@final
class DefaultTargetGenerator:
    """Default target generator: moves control qubits to CZ blockade partners."""

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

@final
class MultiSolveResult:
    """Result of a multi-candidate solve via ``MoveSolver.solve_with_generator()``."""

    @property
    def status(self) -> str:
        """Status of the winning solve."""
        ...

    @property
    def candidate_index(self) -> int | None:
        """Index of the winning candidate, or None if all failed."""
        ...

    @property
    def total_expansions(self) -> int:
        """Total nodes expanded across all candidates."""
        ...

    @property
    def candidates_tried(self) -> int:
        """Number of candidates attempted."""
        ...

    @property
    def attempts(self) -> list[dict[str, object]]:
        """Per-candidate attempt details."""
        ...

    @property
    def move_layers(self) -> list[list[tuple[int, int, int, int, int, int]]]:
        """Move layers from the winning candidate."""
        ...

    @property
    def goal_config(self) -> dict[int, LocationAddress]:
        """Goal configuration from the winning candidate."""
        ...

    @property
    def cost(self) -> float:
        """Path cost from the winning candidate."""
        ...

    @property
    def deadlocks(self) -> int:
        """Deadlocks from the winning candidate."""
        ...

    def __repr__(self) -> str: ...

# ── AtomStateData ──

@final
class AtomStateData:
    """Tracks qubit-to-location mappings as atoms move through the architecture.

    Immutable value type backed by a Rust implementation. Used by the IR
    analysis pipeline to simulate atom movement, detect collisions, and
    identify CZ gate pairings.

    All mutation methods (``add_atoms``, ``apply_moves``) return new instances.
    The two primary maps are kept as a bidirectional index: given a location
    you can find the qubit, and given a qubit you can find the location. When
    a move causes two atoms to occupy the same site, both are removed from the
    location maps and recorded in ``collision``.

    All integer arguments are validated to fit in u32 range (0 to 2^32 - 1).

    Args:
        locations_to_qubit (Optional[dict[LocationAddress, int]]): Reverse index
            from location to qubit id, default = None (empty).
        qubit_to_locations (Optional[dict[int, LocationAddress]]): Forward index
            from qubit id to location, default = None (empty).
        collision (Optional[dict[int, int]]): Cumulative collision record -- key is
            the moving qubit, value is the qubit it displaced, default = None (empty).
        prev_lanes (Optional[dict[int, LaneAddress]]): Lane each qubit used in
            the most recent move step, default = None (empty).
        move_count (Optional[dict[int, int]]): Cumulative move count per qubit,
            default = None (empty).
    """

    def __init__(
        self,
        locations_to_qubit: Optional[dict[LocationAddress, int]] = None,
        qubit_to_locations: Optional[dict[int, LocationAddress]] = None,
        collision: Optional[dict[int, int]] = None,
        prev_lanes: Optional[dict[int, LaneAddress]] = None,
        move_count: Optional[dict[int, int]] = None,
    ) -> None: ...
    @staticmethod
    def from_qubit_locations(locations: dict[int, LocationAddress]) -> AtomStateData:
        """Create a state from a mapping of qubit ids to locations.

        Builds both forward and reverse location maps. Collision, prev_lanes,
        and move_count are initialized to empty.

        Args:
            locations (dict[int, LocationAddress]): Mapping from qubit id to
                its initial location.

        Returns:
            AtomStateData: A new state with the given qubit placements.

        Raises:
            ValueError: If any qubit id is negative or exceeds u32 max.
        """
        ...

    @staticmethod
    def from_location_list(locations: list[LocationAddress]) -> AtomStateData:
        """Create a state from an ordered list of locations.

        Qubit ids are assigned sequentially starting from 0 based on list
        position (i.e. ``locations[0]`` gets qubit 0, ``locations[1]`` gets
        qubit 1, etc.).

        Args:
            locations (list[LocationAddress]): Ordered list of initial qubit
                locations.

        Returns:
            AtomStateData: A new state with sequential qubit ids.
        """
        ...

    @property
    def locations_to_qubit(self) -> dict[LocationAddress, int]:
        """Reverse index: location to qubit id occupying that site."""
        ...

    @property
    def qubit_to_locations(self) -> dict[int, LocationAddress]:
        """Forward index: qubit id to current physical location."""
        ...

    @property
    def collision(self) -> dict[int, int]:
        """Cumulative record of qubit collisions from ``apply_moves`` calls.

        Entries persist across successive ``apply_moves`` calls and are only
        cleared by constructors or ``add_atoms``. Key is the moving qubit id,
        value is the qubit id it displaced. Both qubits are removed from the
        location maps when a collision occurs.
        """
        ...

    @property
    def prev_lanes(self) -> dict[int, LaneAddress]:
        """Lane used by each qubit in the most recent ``apply_moves`` call.

        Only contains entries for qubits that actually moved in the last step.
        """
        ...

    @property
    def move_count(self) -> dict[int, int]:
        """Cumulative move count for each qubit across all ``apply_moves`` calls."""
        ...

    def add_atoms(self, locations: dict[int, LocationAddress]) -> AtomStateData:
        """Add atoms at new locations, returning a new state.

        The new state inherits the current location maps plus the new atoms.
        Collision, prev_lanes, and move_count are reset to empty.

        Args:
            locations (dict[int, LocationAddress]): Mapping from qubit id to
                location for the new atoms.

        Returns:
            AtomStateData: A new state with the additional atoms placed.

        Raises:
            ValueError: If any qubit id is negative or exceeds u32 max.
            RuntimeError: If a qubit id already exists in this state or a
                location is already occupied.
        """
        ...

    def apply_moves(
        self, lanes: list[LaneAddress], arch_spec: ArchSpec
    ) -> Optional[AtomStateData]:
        """Apply a sequence of lane moves and return the resulting state.

        Each lane is resolved to source/destination locations via the arch
        spec. Qubits at source locations are moved to their destinations.
        If a destination is already occupied, both qubits are recorded as
        collided and removed from the location maps. Lanes whose source
        location has no qubit are silently skipped.

        Args:
            lanes (list[LaneAddress]): Sequence of lane addresses to apply.
            arch_spec (ArchSpec): Architecture specification for resolving
                lane endpoints.

        Returns:
            Optional[AtomStateData]: A new state reflecting the moves, or
                ``None`` if any lane address is invalid.
        """
        ...

    def get_qubit(self, location: LocationAddress) -> Optional[int]:
        """Look up which qubit (if any) occupies the given location.

        Args:
            location (LocationAddress): The physical location to query.

        Returns:
            Optional[int]: The qubit id at that location, or ``None`` if empty.
        """
        ...

    def get_qubit_pairing(
        self, zone_address: ZoneAddress, arch_spec: ArchSpec
    ) -> Optional[tuple[list[int], list[int], list[int]]]:
        """Find CZ gate control/target qubit pairings within a zone.

        For each qubit in the zone, checks whether the CZ pair site (via
        the arch spec's entangling zone pairs) is also occupied. If both
        sites have qubits, they form a control/target pair.

        Args:
            zone_address (ZoneAddress): The zone to search for pairings.
            arch_spec (ArchSpec): Architecture specification with CZ pair data.

        Returns:
            Optional[tuple[list[int], list[int], list[int]]]: A tuple
                ``(controls, targets, unpaired)`` where ``controls[i]`` and
                ``targets[i]`` are paired for a CZ gate, and ``unpaired``
                contains qubits whose pair site is empty or doesn't exist.
                Results are sorted by qubit id. Returns ``None`` if the zone
                address is invalid.
        """
        ...

    def copy(self) -> AtomStateData:
        """Return a shallow copy of this state.

        Returns:
            AtomStateData: A copy with identical field values.
        """
        ...

    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

# ── Instruction ──

@final
class Instruction:
    """A single bytecode instruction.

    Instructions are created via static factory methods -- there is no
    direct constructor. All instances are immutable.

    Instruction categories:

    - **Constants**: Push typed values onto the stack.
    - **Stack**: Manipulate the operand stack (pop, dup, swap).
    - **Atom ops**: Fill sites and move atoms (initial_fill, fill, move).
    - **Gates**: Quantum gate operations (local_r, local_rz, global_r, global_rz, cz).
    - **Measurement**: Measure atoms and await results.
    - **Arrays**: Construct and index arrays (new_array, get_item).
    - **Data**: Build detector and observable records.
    - **Control flow**: Return and halt.
    """

    # -- Constants --

    @staticmethod
    def const_float(value: float) -> Instruction:
        """Push a 64-bit float constant onto the stack.

        Args:
            value (float): The float value to push.

        Returns:
            Instruction: The constant instruction.
        """
        ...

    @staticmethod
    def const_int(value: int) -> Instruction:
        """Push a 64-bit signed integer constant onto the stack.

        Args:
            value (int): The signed integer value to push.

        Returns:
            Instruction: The constant instruction.
        """
        ...

    @staticmethod
    def const_loc(zone_id: int, word_id: int, site_id: int) -> Instruction:
        """Push a location address constant onto the stack.

        Args:
            zone_id (int): Zone identifier (0..255).
            word_id (int): Word identifier (0..65535).
            site_id (int): Site identifier (0..65535).

        Returns:
            Instruction: The constant instruction.
        """
        ...

    @staticmethod
    def const_lane(
        move_type: MoveType,
        zone_id: int,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = ...,
    ) -> Instruction:
        """Push a lane address constant onto the stack.

        Args:
            move_type (MoveType): SITE, WORD, or ZONE.
            zone_id (int): Zone identifier (0..255).
            word_id (int): Word identifier (0..65535).
            site_id (int): Site identifier (0..65535).
            bus_id (int): Bus identifier (0..65535).
            direction (Direction): FORWARD or BACKWARD. Default: FORWARD.

        Returns:
            Instruction: The constant instruction.
        """
        ...

    @staticmethod
    def const_zone(zone_id: int) -> Instruction:
        """Push a zone address constant onto the stack.

        Args:
            zone_id (int): Zone identifier (0..255).

        Returns:
            Instruction: The constant instruction.
        """
        ...
    # -- Stack manipulation --

    @staticmethod
    def pop() -> Instruction:
        """Pop and discard the top stack value.

        Returns:
            Instruction: The pop instruction.
        """
        ...

    @staticmethod
    def dup() -> Instruction:
        """Duplicate the top stack value.

        Returns:
            Instruction: The dup instruction.
        """
        ...

    @staticmethod
    def swap() -> Instruction:
        """Swap the top two stack values.

        Returns:
            Instruction: The swap instruction.
        """
        ...
    # -- Atom operations --

    @staticmethod
    def initial_fill(arity: int) -> Instruction:
        """Initial atom fill. Must be the first non-constant instruction.

        Pops ``arity`` location addresses from the stack.

        Args:
            arity (int): Number of location addresses to pop.

        Returns:
            Instruction: The initial_fill instruction.
        """
        ...

    @staticmethod
    def fill(arity: int) -> Instruction:
        """Fill atom sites.

        Pops ``arity`` location addresses from the stack.

        Args:
            arity (int): Number of location addresses to pop.

        Returns:
            Instruction: The fill instruction.
        """
        ...

    @staticmethod
    def move_(arity: int) -> Instruction:
        """Move atoms along lanes.

        Pops ``arity`` lane addresses from the stack.

        Args:
            arity (int): Number of lane addresses to pop.

        Returns:
            Instruction: The move instruction.

        Note: Named ``move_`` to avoid shadowing the Python builtin.
        """
        ...
    # -- Gate operations --

    @staticmethod
    def local_r(arity: int) -> Instruction:
        """Local R rotation gate on ``arity`` locations.

        Pops phi (axis angle) and theta (rotation angle), then ``arity`` locations from the stack.

        Args:
            arity (int): Number of locations to apply the gate to.

        Returns:
            Instruction: The local_r instruction.
        """
        ...

    @staticmethod
    def local_rz(arity: int) -> Instruction:
        """Local Rz rotation gate on ``arity`` locations.

        Pops theta (rotation angle), then ``arity`` locations from the stack.

        Args:
            arity (int): Number of locations to apply the gate to.

        Returns:
            Instruction: The local_rz instruction.
        """
        ...

    @staticmethod
    def global_r() -> Instruction:
        """Global R rotation gate.

        Pops phi (axis angle) and theta (rotation angle) from the stack.

        Returns:
            Instruction: The global_r instruction.
        """
        ...

    @staticmethod
    def global_rz() -> Instruction:
        """Global Rz rotation gate.

        Pops theta (rotation angle) from the stack.

        Returns:
            Instruction: The global_rz instruction.
        """
        ...

    @staticmethod
    def cz() -> Instruction:
        """CZ entangling gate.

        Pops a zone address from the stack.

        Returns:
            Instruction: The cz instruction.
        """
        ...
    # -- Measurement --

    @staticmethod
    def measure(arity: int) -> Instruction:
        """Measure atoms in ``arity`` zones.

        Pops ``arity`` zone addresses from the stack; pushes ``arity``
        measure futures (one per zone).

        Args:
            arity (int): Number of zones to measure.

        Returns:
            Instruction: The measure instruction.
        """
        ...

    @staticmethod
    def await_measure() -> Instruction:
        """Block until the most recent measurement completes.

        Pops one measurement future from the stack (linear consumption)
        and pushes one array reference holding the resolved measurement
        results.

        Returns:
            Instruction: The await_measure instruction.
        """
        ...
    # -- Array operations --

    @staticmethod
    def new_array(type_tag: int, dim0: int, dim1: int = 0) -> Instruction:
        """Create a new array.

        Pops ``dim0 * max(dim1, 1)`` values of any type from the stack
        (the array's initial elements) and pushes a new array reference.

        Args:
            type_tag (int): Element type tag.
            dim0 (int): First dimension size (must be > 0).
            dim1 (int): Second dimension size, default = 0 (1-D array).

        Returns:
            Instruction: The new_array instruction.
        """
        ...

    @staticmethod
    def get_item(ndims: int) -> Instruction:
        """Index into an array.

        Pops ``ndims`` index values then the array from the stack.

        Args:
            ndims (int): Number of index dimensions to pop.

        Returns:
            Instruction: The get_item instruction.
        """
        ...
    # -- Data construction --

    @staticmethod
    def set_detector() -> Instruction:
        """Build a detector record from the top-of-stack array.

        Pops one array reference from the stack and pushes one
        detector reference.

        Returns:
            Instruction: The set_detector instruction.
        """
        ...

    @staticmethod
    def set_observable() -> Instruction:
        """Build an observable record from the top-of-stack array.

        Pops one array reference from the stack and pushes one
        observable reference.

        Returns:
            Instruction: The set_observable instruction.
        """
        ...
    # -- Control flow --

    @staticmethod
    def return_() -> Instruction:
        """Return from the current program.

        Pops one value of any type from the stack as the return value.

        Returns:
            Instruction: The return instruction.

        Note: Named ``return_`` to avoid shadowing the Python keyword.
        """
        ...

    @staticmethod
    def halt() -> Instruction:
        """Halt execution.

        Returns:
            Instruction: The halt instruction.
        """
        ...
    # -- Instance members --

    @property
    def opcode(self) -> int:
        """Packed 16-bit opcode: ``(instruction_code << 8) | device_code``."""
        ...

    def op_name(self) -> str:
        """Lowercase snake_case opcode name matching the bytecode text-format
        parser's canonical names (see
        ``crates/bloqade-lanes-bytecode-core/src/bytecode/text.rs``).

        Factory methods use trailing underscores for Python-keyword conflicts
        (``Instruction.move_()``, ``Instruction.return_()``), but ``op_name``
        returns the parser-canonical bare names: ``"move"`` and ``"return"``.
        """
        ...

    def arity(self) -> int:
        """Arity field for opcodes that carry one.

        Valid on ``initial_fill``, ``fill``, ``move``, ``local_r``,
        ``local_rz``, ``measure``.

        Raises:
            RuntimeError: If called on an opcode without an arity field.
        """
        ...

    def float_value(self) -> float:
        """Value attribute of a ``const_float`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def int_value(self) -> int:
        """Value attribute of a ``const_int`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def location_address(self) -> LocationAddress:
        """Decoded address of a ``const_loc`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def lane_address(self) -> LaneAddress:
        """Decoded address of a ``const_lane`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def zone_address(self) -> ZoneAddress:
        """Decoded address of a ``const_zone`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def type_tag(self) -> int:
        """Type tag attribute of a ``new_array`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def dim0(self) -> int:
        """First dimension of a ``new_array`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def dim1(self) -> int:
        """Second dimension of a ``new_array`` instruction (0 for 1-D).

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def ndims(self) -> int:
        """Number of index dimensions of a ``get_item`` instruction.

        Raises:
            RuntimeError: If called on any other opcode.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

# ── Program ──

@final
class Program:
    """A bytecode program consisting of a version and instruction sequence.

    Programs can be constructed directly, parsed from SST text assembly,
    or deserialized from BLQD binary format.

    Args:
        version (tuple[int, int]): Program version as ``(major, minor)``.
        instructions (list[Instruction]): Instructions in execution order.
    """

    def __init__(
        self, version: tuple[int, int], instructions: list[Instruction]
    ) -> None: ...
    @staticmethod
    def from_text(source: str) -> Program:
        """Parse a program from SST text assembly format.

        Args:
            source (str): SST text assembly source.

        Returns:
            Program: The parsed program.

        Raises:
            ParseError: If the source text is malformed.
        """
        ...

    def to_text(self) -> str:
        """Serialize the program to SST text assembly format.

        Returns:
            str: The SST text representation.
        """
        ...

    @staticmethod
    def from_binary(data: bytes) -> Program:
        """Deserialize a program from BLQD binary format.

        Args:
            data (bytes): Raw BLQD binary data.

        Returns:
            Program: The deserialized program.

        Raises:
            ProgramError: If the binary data is malformed.
        """
        ...

    def to_binary(self) -> bytes:
        """Serialize the program to BLQD binary format.

        Returns:
            bytes: The BLQD binary representation.
        """
        ...

    def validate(
        self,
        arch: Optional[ArchSpec] = None,
        stack: bool = False,
    ) -> None:
        """Validate the program.

        Structural validation always runs. With ``arch``, also validates
        addresses and capabilities. With ``stack=True``, runs stack type
        simulation.

        Args:
            arch (Optional[ArchSpec]): Architecture spec for address validation.
            stack (bool): Whether to run stack type simulation. Default: False.

        Raises:
            ValidationError: With ``.errors`` list containing individual
                error subclass instances.
        """
        ...

    @property
    def version(self) -> tuple[int, int]:
        """Program version as ``(major, minor)``."""
        ...

    @property
    def instructions(self) -> list[Instruction]:
        """Instructions in execution order."""
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
