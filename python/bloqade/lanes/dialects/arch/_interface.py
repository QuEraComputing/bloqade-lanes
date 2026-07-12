from kirin import lowering

from bloqade.lanes.bytecode.encoding import LocationAddress

from .stmts import CzPartner, Loc


@lowering.wraps(Loc)
def loc(zone: int, row: int, col: int) -> LocationAddress:
    """Construct a LocationAddress from a ``(zone, row, col)`` grid coordinate.

    ``col`` is the grid x-index and ``row`` the grid y-index within ``zone``;
    the coordinate is resolved to a physical ``LocationAddress`` (``word_id`` +
    ``site_id``) against the architecture's addressing scheme
    (``ArchSpec.location_at``) at compile time. All three must be
    compile-time-constant integers.
    """
    ...


@lowering.wraps(CzPartner)
def cz_partner(address: LocationAddress) -> LocationAddress:
    """Return the CZ blockade-partner location of ``address``.

    An atom placed at the returned location can be CZ-entangled with an atom
    at ``address``. Useful for staging a ``move_to`` onto a partner site
    without hardcoding the arch's word/site layout, e.g.::

        move_to([q], [cz_partner(static_loc)])

    ``address`` must resolve to a compile-time-constant location (the partner
    is looked up in the architecture spec during compilation).
    """
    ...
