from bloqade.geometry.dialects import grid as geometry_grid

from bloqade.lanes.bytecode._native import (
    ArchSpec as ArchSpec,
    Bus as Bus,
    Buses as Buses,
    Geometry as Geometry,
    Grid as Grid,
    TransportPath as TransportPath,
    Word as Word,
    Zone as Zone,
)


def grid_to_rust(g: geometry_grid.Grid) -> Grid:
    """Convert a bloqade-geometry Grid to a Rust bytecode Grid."""
    return Grid(
        x_start=g.x_init,  # type: ignore[arg-type]
        y_start=g.y_init,  # type: ignore[arg-type]
        x_spacing=list(g.x_spacing),
        y_spacing=list(g.y_spacing),
    )


def grid_from_rust(g: Grid) -> geometry_grid.Grid:
    """Convert a Rust bytecode Grid to a bloqade-geometry Grid."""
    return geometry_grid.Grid(
        x_init=g.x_start,
        y_init=g.y_start,
        x_spacing=list(g.x_spacing),  # type: ignore[arg-type]
        y_spacing=list(g.y_spacing),  # type: ignore[arg-type]
    )
