"""Bloqade lanes bytecode compiler and validation framework.

Provides types for constructing, serializing, and validating lane-move
bytecode programs targeting the Bloqade quantum computing platform.

Core types:
    - :class:`Program` -- bytecode program (construct, parse, serialize, validate)
    - :class:`Instruction` -- individual bytecode instruction (factory methods)
    - :class:`ArchSpec` -- device architecture specification

Address types:
    - :class:`LocationAddress` -- bit-packed atom location (zone + word + site)
    - :class:`LaneAddress` -- bit-packed lane address (direction, move type, zone, word, site, bus)
    - :class:`ZoneAddress` -- bit-packed zone address

Architecture building blocks:
    - :class:`Word`, :class:`Grid`
    - :class:`SiteBus`, :class:`WordBus`, :class:`ZoneBus`
    - :class:`Zone`, :class:`Mode`, :class:`TransportPath`

Enums:
    - :class:`Direction` -- FORWARD / BACKWARD
    - :class:`MoveType` -- SITE / WORD / ZONE

C library helpers:
    - :func:`has_clib`, :func:`include_dir`, :func:`lib_dir`, :func:`lib_path`

Exception hierarchy:
    - :class:`ArchSpecError` -- architecture validation (18 subclasses)
    - :class:`ValidationError` -- bytecode validation
    - :class:`ParseError` -- SST text format parsing
    - :class:`ProgramError` -- BLQD binary format parsing
    - :class:`DecodeError` -- instruction decoding
"""

from bloqade.lanes.bytecode._clib_path import (
    has_clib as has_clib,
    include_dir as include_dir,
    lib_dir as lib_dir,
    lib_path as lib_path,
)
from bloqade.lanes.bytecode._native import (
    ArchSpec as ArchSpec,
    AtomStateData as AtomStateData,
    Direction as Direction,
    Grid as Grid,
    Instruction as Instruction,
    LaneAddress as LaneAddress,
    LocationAddress as LocationAddress,
    Mode as Mode,
    MoveSolver as MoveSolver,
    MoveType as MoveType,
    Program as Program,
    SearchStrategy as SearchStrategy,
    SiteBus as SiteBus,
    SolveResult as SolveResult,
    TransportPath as TransportPath,
    Word as Word,
    WordBus as WordBus,
    Zone as Zone,
    ZoneAddress as ZoneAddress,
    ZoneBus as ZoneBus,
)
from bloqade.lanes.bytecode.exceptions import (
    ArchSpecError as ArchSpecError,
    DecodeError as DecodeError,
    ParseError as ParseError,
    ProgramError as ProgramError,
    ValidationError as ValidationError,
)
