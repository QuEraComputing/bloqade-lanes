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
    """

    SITE: MoveType
    WORD: MoveType
    @property
    def name(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...

# ── Address Types ──

@final
class LocationAddress:
    """Bit-packed atom location address.

    Encodes a physical atom location as ``word_id`` (16 bits) and
    ``site_id`` (16 bits) into a 32-bit word.

    Layout: ``[word_id:16][site_id:16]``

    Args:
        word_id (int): Word identifier (0..65535).
        site_id (int): Site identifier within the word (0..65535).
    """

    def __init__(self, word_id: int, site_id: int) -> None: ...
    @property
    def word_id(self) -> int:
        """Word identifier."""
        ...

    @property
    def site_id(self) -> int:
        """Site identifier within the word."""
        ...

    def encode(self) -> int:
        """Encode to a 32-bit packed integer.

        Returns:
            int: The 32-bit packed representation.
        """
        ...

    @staticmethod
    def decode(bits: int) -> LocationAddress:
        """Decode a 32-bit packed integer into a LocationAddress.

        Args:
            bits (int): The 32-bit packed representation.

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

    Encodes direction (1 bit), move_type (1 bit), word_id (16 bits),
    site_id (16 bits), and bus_id (16 bits) across two 32-bit data words,
    returned as a combined 64-bit value.

    Layout:
        data0: ``[word_id:16][site_id:16]``
        data1: ``[dir:1][mt:1][pad:14][bus_id:16]``

    Args:
        move_type (MoveType): SITE or WORD.
        word_id (int): Word identifier (0..65535).
        site_id (int): Site identifier within the word (0..65535).
        bus_id (int): Bus identifier (0..65535).
        direction (Direction): Forward or Backward. Default: Direction.FORWARD.
    """

    def __init__(
        self,
        move_type: MoveType,
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
        """Bus type (SITE or WORD)."""
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

    Encodes a zone identifier (16 bits) into a 32-bit value.

    Layout: ``[pad:16][zone_id:16]``

    Args:
        zone_id (int): Zone identifier (0..65535).
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

    Each word contains a fixed number of sites (determined by
    ``Geometry.sites_per_word``). Sites are positioned on the word's
    grid via ``(x_idx, y_idx)`` index pairs.

    Args:
        positions (Grid): Coordinate grid for this word's sites.
        site_indices (list[tuple[int, int]]): Site positions as ``(x_idx, y_idx)`` grid index pairs.

    Note: A word's identity is determined by its position in the ``Geometry.words`` list.
    """

    def __init__(
        self,
        positions: Grid,
        site_indices: list[tuple[int, int]],
    ) -> None: ...
    @property
    def positions(self) -> Grid:
        """Coordinate grid for this word's sites."""
        ...

    @property
    def site_indices(self) -> list[tuple[int, int]]:
        """Site positions as ``(x_idx, y_idx)`` grid index pairs."""
        ...

    def site_position(self, site_idx: int) -> Optional[tuple[float, float]]:
        """Look up the ``(x, y)`` physical position of a site by index.

        Args:
            site_idx (int): Index of the site within this word.

        Returns:
            tuple[float, float]: The ``(x, y)`` physical position, or None if out of range.
        """
        ...

    def __repr__(self) -> str: ...

@final
class Bus:
    """A transport bus that maps source positions to destination positions.

    The ``src`` and ``dst`` lists are parallel arrays: ``src[i]`` maps to
    ``dst[i]``. For site buses, values are site indices within a word. For
    word buses, values are word IDs. Whether a bus is a site bus or word bus
    is determined by which list it belongs to in ``Buses``.

    Args:
        src (list[int]): Source indices (site indices for site buses, word IDs for word buses).
        dst (list[int]): Destination indices.

    Note: A bus's identity is determined by its position in the parent list.
    """

    def __init__(
        self,
        src: list[int],
        dst: list[int],
        words: list[int] | None = None,
    ) -> None: ...
    @property
    def src(self) -> list[int]:
        """Source indices."""
        ...

    @property
    def dst(self) -> list[int]:
        """Destination indices."""
        ...

    @property
    def words(self) -> list[int] | None:
        """Word IDs this bus applies to (site buses only). None = global fallback."""
        ...

    def resolve_forward(self, src: int) -> Optional[int]:
        """Map a source value to its destination.

        Args:
            src (int): Source index to look up.

        Returns:
            int: The corresponding destination index, or None if not found.
        """
        ...

    def resolve_backward(self, dst: int) -> Optional[int]:
        """Map a destination value back to its source.

        Args:
            dst (int): Destination index to look up.

        Returns:
            int: The corresponding source index, or None if not found.
        """
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Buses:
    """Container for all site buses and word buses in an architecture.

    Args:
        site_buses (list[Bus]): Site bus definitions.
        word_buses (list[Bus]): Word bus definitions.
    """

    def __init__(self, site_buses: list[Bus], word_buses: list[Bus]) -> None: ...
    @property
    def site_buses(self) -> list[Bus]:
        """All site bus definitions."""
        ...

    @property
    def word_buses(self) -> list[Bus]:
        """All word bus definitions."""
        ...

    def __repr__(self) -> str: ...

@final
class Zone:
    """A group of words that form a logical zone.

    Zones partition words for operations like entangling gates and
    measurement. Zone 0 must contain all words.

    Args:
        words (list[int]): Word identifiers belonging to this zone.

    Note: A zone's identity is determined by its position in the ``ArchSpec.zones`` list.
    """

    def __init__(self, words: list[int]) -> None: ...
    @property
    def words(self) -> list[int]:
        """Word identifiers belonging to this zone."""
        ...

    def __repr__(self) -> str: ...

@final
class Geometry:
    """Device geometry: the set of words and their site layout.

    Args:
        sites_per_word (int): Number of atom sites in each word.
        words (list[Word]): Word definitions.
    """

    def __init__(self, sites_per_word: int, words: list[Word]) -> None: ...
    @property
    def sites_per_word(self) -> int:
        """Number of atom sites in each word."""
        ...

    @property
    def words(self) -> list[Word]:
        """All word definitions."""
        ...

    def __repr__(self) -> str: ...

@final
class TransportPath:
    """A transport path for a lane, defined by waypoints.

    The lane is identified by a ``LaneAddress`` which encodes the direction,
    move type, word, site, and bus.

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

    Describes the full hardware topology: geometry (words, sites, grids),
    bus connectivity, zones, and operational constraints. Can be loaded
    from JSON or constructed programmatically.

    Args:
        version (tuple[int, int]): Spec version as ``(major, minor)``.
        geometry (Geometry): Device geometry (words and sites).
        buses (Buses): Bus connectivity (site buses and word buses).
        words_with_site_buses (list[int]): Word IDs that participate in site-bus moves.
        sites_with_word_buses (list[int]): Site indices that participate in word-bus moves.
        zones (list[Zone]): Zone definitions partitioning words.
        entangling_zones (list[list[tuple[int, int]]]): Entangling zones as lists of word-ID pairs.
        measurement_mode_zones (list[int]): Zone IDs for measurement (first must be zone 0).
        paths (Optional[list[TransportPath]]): AOD transport paths, default = None.
        feed_forward (bool): Whether the device supports mid-circuit measurement with classical feedback. Default = False.
        atom_reloading (bool): Whether the device supports reloading atoms after initial fill. Default = False.
        blockade_radius (float): Rydberg blockade radius in micrometers. Default = 2.0.
    """

    def __init__(
        self,
        version: tuple[int, int],
        geometry: Geometry,
        buses: Buses,
        words_with_site_buses: list[int],
        sites_with_word_buses: list[int],
        zones: list[Zone],
        entangling_zones: list[list[tuple[int, int]]],
        measurement_mode_zones: list[int],
        paths: Optional[list[TransportPath]] = None,
        feed_forward: bool = False,
        atom_reloading: bool = False,
        blockade_radius: float = 2.0,
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
    def geometry(self) -> Geometry:
        """Device geometry."""
        ...

    @property
    def buses(self) -> Buses:
        """Bus connectivity."""
        ...

    @property
    def words_with_site_buses(self) -> list[int]:
        """Word IDs that participate in site-bus moves."""
        ...

    @property
    def sites_with_word_buses(self) -> list[int]:
        """Site indices that participate in word-bus moves."""
        ...

    @property
    def zones(self) -> list[Zone]:
        """Zone definitions."""
        ...

    @property
    def entangling_zones(self) -> list[list[tuple[int, int]]]:
        """Entangling zones as lists of word-ID pairs."""
        ...

    @property
    def blockade_radius(self) -> float:
        """Rydberg blockade radius in micrometers."""
        ...

    @property
    def measurement_mode_zones(self) -> list[int]:
        """Zone IDs for measurement mode."""
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
    def paths(self) -> Optional[list[TransportPath]]:
        """Transport paths between locations, or None."""
        ...

    def word_by_id(self, id: int) -> Optional[Word]:
        """Look up a word by its index.

        Args:
            id (int): Word index in ``geometry.words``.

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

    def site_bus_by_id(self, id: int) -> Optional[Bus]:
        """Look up a site bus by its index.

        Args:
            id (int): Site bus index in ``buses.site_buses``.

        Returns:
            Bus: The site bus, or None if not found.
        """
        ...

    def word_bus_by_id(self, id: int) -> Optional[Bus]:
        """Look up a word bus by its index.

        Args:
            id (int): Word bus index in ``buses.word_buses``.

        Returns:
            Bus: The word bus, or None if not found.
        """
        ...

    def location_position(self, loc: LocationAddress) -> Optional[tuple[float, float]]:
        """Get the ``(x, y)`` physical position for an atom location.

        Args:
            loc (LocationAddress): The location address to look up.

        Returns:
            tuple[float, float]: The ``(x, y)`` position, or None if the word or site
                is not found.
        """
        ...

    def lane_endpoints(
        self, lane: LaneAddress
    ) -> Optional[tuple[LocationAddress, LocationAddress]]:
        """Resolve a lane address to its source and destination locations.

        Traces through the appropriate bus (site bus or word bus) in the
        specified direction (forward or backward) to determine which two
        ``LocationAddress`` endpoints the lane connects.

        Args:
            lane (LaneAddress): The lane address to resolve.

        Returns:
            tuple[LocationAddress, LocationAddress]: A ``(src, dst)`` pair, or None if the
                lane references an invalid bus, word, or site.
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

        Checks for duplicate addresses and invalid word/site combinations.

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
    def __hash__(self) -> int: ...

# ── Move Solver ──

@final
class SolveResult:
    """Result of a move synthesis solve."""

    @property
    def move_layers(self) -> list[list[tuple[int, int, int, int, int]]]:
        """Move layers as lists of (direction, move_type, word_id, site_id, bus_id) tuples."""
        ...

    @property
    def goal_config(self) -> list[tuple[int, int, int]]:
        """Goal configuration as (qubit_id, word_id, site_id) tuples."""
        ...

    @property
    def nodes_expanded(self) -> int:
        """Number of nodes expanded during search."""
        ...

    @property
    def cost(self) -> float:
        """Total path cost."""
        ...

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
        initial: list[tuple[int, int, int]],
        target: list[tuple[int, int, int]],
        blocked: list[tuple[int, int]],
        max_expansions: Optional[int] = None,
        strategy: str = "astar",
        top_c: int = 3,
        max_movesets_per_group: int = 3,
        weight: float = 1.0,
        mobility_weight: float = 0.0,
    ) -> Optional[SolveResult]:
        """Solve a move synthesis problem.

        Args:
            initial: List of (qubit_id, word_id, site_id) starting positions.
            target: List of (qubit_id, word_id, site_id) desired positions.
            blocked: List of (word_id, site_id) immovable obstacle locations.
            max_expansions: Optional limit on node expansions.
            strategy: Search strategy: "astar", "dfs", "bfs", "greedy", "ids".
            top_c: Top bus options per qubit in the heuristic expander.
            max_movesets_per_group: Max movesets per bus group.
            weight: Heuristic weight for A* (1.0 = standard, >1.0 = bounded suboptimal).
            mobility_weight: Weight for mobility bonus in expander scoring (0.0 = disabled).

        Returns:
            SolveResult if a solution is found, None otherwise.
        """
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
        collision (Optional[dict[int, int]]): Cumulative collision record — key is
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
        the arch spec's blockaded location data) is also occupied. If both
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
    def const_loc(word_id: int, site_id: int) -> Instruction:
        """Push a location address constant onto the stack.

        Args:
            word_id (int): Word identifier (0..255).
            site_id (int): Site identifier (0..255).

        Returns:
            Instruction: The constant instruction.
        """
        ...

    @staticmethod
    def const_lane(
        move_type: MoveType,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = ...,
    ) -> Instruction:
        """Push a lane address constant onto the stack.

        Args:
            move_type (MoveType): SITE or WORD.
            word_id (int): Word identifier (0..255).
            site_id (int): Site identifier (0..255).
            bus_id (int): Bus identifier (0..255).
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
        """Measure atoms at ``arity`` locations.

        Pops ``arity`` location addresses from the stack.

        Args:
            arity (int): Number of locations to measure.

        Returns:
            Instruction: The measure instruction.
        """
        ...

    @staticmethod
    def await_measure() -> Instruction:
        """Block until the most recent measurement completes.

        Returns:
            Instruction: The await_measure instruction.
        """
        ...
    # -- Array operations --

    @staticmethod
    def new_array(type_tag: int, dim0: int, dim1: int = 0) -> Instruction:
        """Create a new array.

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

        Returns:
            Instruction: The set_detector instruction.
        """
        ...

    @staticmethod
    def set_observable() -> Instruction:
        """Build an observable record from the top-of-stack array.

        Returns:
            Instruction: The set_observable instruction.
        """
        ...
    # -- Control flow --

    @staticmethod
    def return_() -> Instruction:
        """Return from the current program.

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

    def validate(self, arch: Optional[ArchSpec] = None, stack: bool = False) -> None:
        """Validate the program.

        Validation runs in layers:
        - **Structural** (always): operand bounds, arity limits, instruction ordering.
        - **Architecture** (when ``arch`` is provided): address validity against the device spec.
        - **Stack simulation** (when ``stack=True``): type checking via abstract interpretation.

        All errors are collected before raising.

        Args:
            arch (Optional[ArchSpec]): Architecture spec for address validation, default = None.
            stack (bool): If True, run stack type simulation (uses ``arch`` if provided),
                default = False.

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
    def __len__(self) -> int:
        """Number of instructions in the program."""
        ...

    def __eq__(self, other: object) -> bool: ...
