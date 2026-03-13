"""Helpers to locate the bundled C library and headers.

When the Python package is installed from a wheel built with
``just build-wheel``, the C shared/static library and header are
installed into ``sys.prefix``:

- ``sys.prefix/lib/`` — shared and static libraries
- ``sys.prefix/include/`` — C header file

Usage::

    from bloqade.lanes.bytecode import clib_dir, include_dir, lib_dir

    # Pass to a C compiler:
    #   -I{include_dir()}  -L{lib_dir()}  -lbloqade_lanes_bytecode
"""

from __future__ import annotations

import sys
from pathlib import Path


def include_dir() -> Path:
    """Directory containing ``bloqade_lanes_bytecode.h``."""
    return Path(sys.prefix) / "include"


def lib_dir() -> Path:
    """Directory containing the shared and static libraries."""
    return Path(sys.prefix) / "lib"


def lib_path() -> Path:
    """Path to the shared library file for the current platform."""
    lib = lib_dir()
    if sys.platform == "darwin":
        return lib / "libbloqade_lanes_bytecode.dylib"
    elif sys.platform == "win32":
        return lib / "bloqade_lanes_bytecode.dll"
    else:
        return lib / "libbloqade_lanes_bytecode.so"


def has_clib() -> bool:
    """Return True if the C library artifacts are bundled."""
    return lib_path().exists()
