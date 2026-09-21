"""Bridge ``move.LogicalInitialize`` to Gemini state-injection kernel arguments.

On Gemini hardware a logical qubit is prepared by *state injection*: a single
physical qubit is rotated into the requested state and then encoded into the
Steane [[7,1,3]] code by a fixed pulse schedule. The pulse kernel that does
this (``state_injection_init`` in ``qlue``) is parameterised by, for each of
the two interleaved column groups of the Gemini grid, a ``y_mask`` selecting
the active rows plus per-row ``axis_angle`` / ``rotation_angle`` lists, and by
one shared CZ top-hat window.

Everything the bridge needs to know is a lanes convention:

* the angle units of ``move.LogicalInitialize`` (**turns**, ``1.0 == 2*pi``),
* how a :class:`~bloqade.lanes.bytecode.encoding.LocationAddress` maps onto
  the ``(col_group, row)`` geometry of the Gemini arch spec,
* the shape of the ``y_mask`` and per-row angle lists.

Keeping the adapter here, next to the dialect it consumes, means it moves in
lock-step with those conventions instead of drifting in a downstream repo.

Angle convention
----------------
``move.LogicalInitialize`` carries the ``U3(theta, phi, lam)`` Euler angles that
prepare each logical qubit from ``|0>``. Matching

    U3(theta, phi, lam)|0> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>

against a rotation ``R_n(beta)`` about an equatorial axis ``n`` at angle
``alpha`` from X,

    R_n(beta)|0> = cos(beta/2)|0> - i e^{i alpha} sin(beta/2)|1>,

gives ``beta = theta`` and ``alpha = phi + pi/2``. In turns that is
``rotation_angle = theta`` and ``axis_angle = phi + 0.25``. ``lam`` only
contributes a global phase on ``|0>`` and is dropped. This is the inverse of
the ``axis_angle -> (phi, lam)`` mapping ``move2squin`` applies to
``move.LocalR`` / ``move.GlobalR``, so both directions agree.

The CZ top-hat window is a hardware calibration quantity expressed in the
pulse kernels' own coordinate frame (which is *not* the arch spec's
corner-at-zero frame), so it is always supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from kirin import ir, types
from kirin.dialects import ilist, py

from bloqade.lanes.arch.geometry import ArchSpecGeometry
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import move

N_COL_GROUPS = 2
"""Number of interleaved column groups on the Gemini grid."""

N_ROWS = 5
"""Number of logical-qubit rows per column group on the Gemini grid."""

QUARTER_TURN = 0.25
"""Offset (turns) from the ``phi`` Euler angle to the rotation-axis angle."""

_COLUMNS_PER_PERIOD = 4
"""Grid columns per repeat of the ``left0, right0, left1, right1`` pattern."""


class CZWindow(NamedTuple):
    """CZ top-hat window in the pulse kernels' coordinate frame (micrometres).

    ``ymin``/``ymax`` bound the rows that should see the CZ pulse;
    ``ymin_keepout``/``ymax_keepout`` extend that by the keep-out margin. The
    window is global in ``x`` and shared by both column groups, so it must
    span every row that is active in either ``y_mask``.
    """

    ymin: float
    ymax: float
    ymin_keepout: float
    ymax_keepout: float

    @classmethod
    def from_bounds(
        cls, ymin: float, ymax: float, keep_out_buffer: float = 10.0
    ) -> CZWindow:
        """Build a window from row bounds plus a symmetric keep-out margin."""
        return cls(ymin, ymax, ymin - keep_out_buffer, ymax + keep_out_buffer)

    def validate(self) -> None:
        if not (self.ymin_keepout <= self.ymin <= self.ymax <= self.ymax_keepout):
            raise ValueError(
                "CZWindow must satisfy ymin_keepout <= ymin <= ymax <= ymax_keepout, "
                f"got {self!r}"
            )


def _match_index(
    positions: Sequence[float], value: float, atol: float, axis: str
) -> int:
    for index, position in enumerate(positions):
        if abs(position - value) <= atol:
            return index
    raise ValueError(f"{axis}={value} does not match any grid {axis}-position")


def resolve_col_group_and_row(
    arch_spec: ArchSpec, location: LocationAddress, *, atol: float = 1e-6
) -> tuple[int, int]:
    """Resolve a location to its Gemini ``(col_group, row)``.

    The Gemini grid repeats a four-column pattern along ``x``: the left and
    right sites of a column-group-0 word pair, then the left and right sites
    of a column-group-1 word pair. ``col_group`` is which of the two
    interleaved groups the location's column falls in (irrespective of
    left/right), and ``row`` is the index of its ``y`` position in the zone
    grid. Works for both the logical and the physical Gemini specs since they
    share this column pattern.

    Raises:
        ValueError: if ``location`` is not a valid site of ``arch_spec`` or
            its position is not on the zone grid.
    """
    x, y = arch_spec.get_position(location)
    grid = ArchSpecGeometry(arch_spec).get_zone_grid(location.zone_id)
    col = _match_index(tuple(grid.x_positions), x, atol, "x")
    row = _match_index(tuple(grid.y_positions), y, atol, "y")
    col_group = (col % _COLUMNS_PER_PERIOD) // (_COLUMNS_PER_PERIOD // N_COL_GROUPS)
    return col_group, row


def theta_phi_to_axis_rotation_angle(
    theta: ir.SSAValue,
    phi: ir.SSAValue,
    insertion_point: ir.Statement,
    *,
    quarter_turn: ir.SSAValue | None = None,
) -> tuple[ir.SSAValue, ir.SSAValue]:
    """Insert IR mapping U3 ``(theta, phi)`` to ``(axis_angle, rotation_angle)``.

    All values are in turns. ``rotation_angle`` is ``theta`` itself (no new
    statement); ``axis_angle`` is ``phi + 0.25``. Pass ``quarter_turn`` to
    reuse an existing ``0.25`` constant instead of inserting a new one.
    """
    if quarter_turn is None:
        (quarter_stmt := py.Constant(QUARTER_TURN)).insert_before(insertion_point)
        quarter_turn = quarter_stmt.result
    (axis_stmt := py.Add(phi, quarter_turn)).insert_before(insertion_point)
    return axis_stmt.result, theta


def logical_initialize_to_state_injection_args(
    node: move.LogicalInitialize,
    arch_spec: ArchSpec,
    cz_window: CZWindow,
    *,
    n_rows: int = N_ROWS,
    atol: float = 1e-6,
) -> tuple[ir.SSAValue, ...]:
    """Build the ``state_injection_init`` argument list for a ``LogicalInitialize``.

    Inserts the supporting statements before ``node`` (which is left in
    place for the caller to replace) and returns, in order::

        (y_mask_col0, axis_angles_col0, rotation_angles_col0,
         y_mask_col1, axis_angles_col1, rotation_angles_col1,
         ymin, ymax, ymin_keepout, ymax_keepout)

    Each ``y_mask`` is an ``IList[bool, n_rows]`` marking the rows of that
    column group that hold a logical qubit; the aligned angle lists carry
    that qubit's angles, or ``0.0`` for rows the mask switches off (inert, as
    the mask removes them from the pulse entirely). Angles are in turns; see
    the module docstring for the ``U3 -> (axis, rotation)`` derivation.

    Args:
        node: the ``move.LogicalInitialize`` to bridge.
        arch_spec: the Gemini arch spec ``node``'s addresses were placed on.
        cz_window: CZ top-hat window in the pulse kernels' frame.
        n_rows: rows per column group expected by the pulse kernel.
        atol: tolerance for matching site positions to the zone grid.

    Raises:
        ValueError: if two addresses resolve to the same ``(col_group, row)``,
            a row is outside ``range(n_rows)``, or the window is malformed.
    """
    cz_window.validate()

    slots: dict[tuple[int, int], LocationAddress] = {}
    angles: dict[tuple[int, int], tuple[ir.SSAValue, ir.SSAValue]] = {}
    axis_by_phi: dict[ir.SSAValue, ir.SSAValue] = {}
    quarter_turn: ir.SSAValue | None = None

    for location, theta, phi in zip(node.location_addresses, node.thetas, node.phis):
        col_group, row = resolve_col_group_and_row(arch_spec, location, atol=atol)
        if row >= n_rows:
            raise ValueError(
                f"{location!r} resolves to row {row}, but the state-injection "
                f"kernel only addresses {n_rows} rows"
            )
        slot = (col_group, row)
        if slot in slots:
            raise ValueError(
                f"{location!r} and {slots[slot]!r} both resolve to column group "
                f"{col_group}, row {row}; a LogicalInitialize may hold at most "
                "one logical qubit per slot"
            )
        slots[slot] = location

        if (axis_angle := axis_by_phi.get(phi)) is None:
            if quarter_turn is None:
                (quarter_stmt := py.Constant(QUARTER_TURN)).insert_before(node)
                quarter_turn = quarter_stmt.result
            axis_angle, _ = theta_phi_to_axis_rotation_angle(
                theta, phi, node, quarter_turn=quarter_turn
            )
            axis_by_phi[phi] = axis_angle
        angles[slot] = (axis_angle, theta)

    zero: ir.SSAValue | None = None

    def filler() -> ir.SSAValue:
        nonlocal zero
        if zero is None:
            (zero_stmt := py.Constant(0.0)).insert_before(node)
            zero = zero_stmt.result
        return zero

    args: list[ir.SSAValue] = []
    for col_group in range(N_COL_GROUPS):
        mask = [(col_group, row) in angles for row in range(n_rows)]
        (mask_stmt := py.Constant(ilist.IList(mask, elem=types.Bool))).insert_before(
            node
        )
        axis_values = tuple(
            angles[(col_group, row)][0] if mask[row] else filler()
            for row in range(n_rows)
        )
        rotation_values = tuple(
            angles[(col_group, row)][1] if mask[row] else filler()
            for row in range(n_rows)
        )
        (axis_stmt := ilist.New(axis_values)).insert_before(node)
        (rotation_stmt := ilist.New(rotation_values)).insert_before(node)
        args.extend((mask_stmt.result, axis_stmt.result, rotation_stmt.result))

    for bound in cz_window:
        (bound_stmt := py.Constant(float(bound))).insert_before(node)
        args.append(bound_stmt.result)

    return tuple(args)


def make_state_injection_args_getter(
    arch_spec: ArchSpec,
    cz_window: CZWindow,
    *,
    n_rows: int = N_ROWS,
    atol: float = 1e-6,
) -> Callable[[move.LogicalInitialize], tuple[ir.SSAValue, ...]]:
    """Bind everything but the node, for ``RewriteLogicalInitialize``-style hooks.

    Downstream rewrites that swap a ``LogicalInitialize`` for a kernel
    invocation take a ``node -> args`` callable; this returns one that calls
    :func:`logical_initialize_to_state_injection_args` with the given spec
    and window.
    """

    def get_args(node: move.LogicalInitialize) -> tuple[ir.SSAValue, ...]:
        return logical_initialize_to_state_injection_args(
            node, arch_spec, cz_window, n_rows=n_rows, atol=atol
        )

    return get_args
