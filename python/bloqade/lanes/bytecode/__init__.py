"""Bloqade lanes bytecode compiler and validation framework.

Provides types for constructing, serializing, and validating lane-move
bytecode programs targeting the Bloqade quantum computing platform.

Core types:
    - :class:`Program` -- bytecode program (construct, parse, serialize, validate)
    - :class:`Instruction` -- individual bytecode instruction (factory methods)
    - :class:`ArchSpec` -- device architecture specification

Address types:
    - :class:`LocationAddr` -- bit-packed atom location (word + site)
    - :class:`LaneAddr` -- bit-packed lane address (direction, move type, word, site, bus)
    - :class:`ZoneAddr` -- bit-packed zone address

Architecture building blocks:
    - :class:`Geometry`, :class:`Word`, :class:`Grid`
    - :class:`Buses`, :class:`Bus`
    - :class:`Zone`, :class:`TransportPath`

Enums:
    - :class:`Direction` -- Forward / Backward
    - :class:`MoveType` -- SiteBus / WordBus

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
    Bus as Bus,
    Buses as Buses,
    Direction as Direction,
    Geometry as Geometry,
    Grid as Grid,
    Instruction as Instruction,
    LaneAddr as LaneAddr,
    LocationAddr as LocationAddr,
    MoveType as MoveType,
    Program as Program,
    TransportPath as TransportPath,
    Word as Word,
    Zone as Zone,
    ZoneAddr as ZoneAddr,
)
from bloqade.lanes.bytecode.exceptions import (
    ArchSpecError as ArchSpecError,
    DecodeError as DecodeError,
    ParseError as ParseError,
    ProgramError as ProgramError,
    ValidationError as ValidationError,
)
