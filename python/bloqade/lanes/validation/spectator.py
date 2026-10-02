"""Spectator policies: what a subroutine frame may do to atoms outside it.

A footprint alone does not isolate a callee: ``move.CZ(zone)`` entangles every
complete pair in the zone, measurement reads every atom in its zones, and global
pulses hit every atom. How strict to be depends on the machine, so the rule is a
policy chosen per compilation. The frame records facts (its ``Effects``); the
policy decides whether they are acceptable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bloqade.lanes.analysis.atom import AtomStateData
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects.qmove import Frame


def format_location(loc: LocationAddress) -> str:
    return f"(zone {loc.zone_id}, word {loc.word_id}, site {loc.site_id})"


class SpectatorPolicy(ABC):
    @abstractmethod
    def check_frame(self, frame: Frame, arch: ArchSpec) -> list[str]:
        """Static: are these effects acceptable for this footprint on this arch?"""

    def check_call(
        self, frame: Frame, atoms: AtomStateData, arch: ArchSpec
    ) -> list[str]:
        """At an invoke, once atom positions exist: are the spectators safe?

        Needs synthesized IR; implemented by the subroutine synthesis work.
        """
        raise NotImplementedError("check_call needs synthesized atom positions")


class ZonedPolicy(SpectatorPolicy):
    """Strict. At a call (``check_call``, not yet implemented), no spectator may
    be anywhere in the frame's ``cz_zones`` or ``measure_zones``."""

    def check_frame(self, frame: Frame, arch: ArchSpec) -> list[str]:
        if frame.effects.global_pulses:
            return ["ZonedPolicy: subroutines may not use global pulses"]
        return []


class SingleZonePolicy(SpectatorPolicy):
    """Permissive. Spectators may share ``cz_zones`` as long as no two of them
    form a complete pair (``check_call``, not yet implemented); the footprint
    must be pair-closed so no callee atom can pair with a spectator. Measuring
    the only zone reads every atom, so subroutines may not measure."""

    def check_frame(self, frame: Frame, arch: ArchSpec) -> list[str]:
        problems = []
        if frame.effects.global_pulses:
            problems.append("SingleZonePolicy: subroutines may not use global pulses")
        if frame.effects.measure_zones:
            problems.append("SingleZonePolicy: subroutines may not measure")
        footprint = frame.footprint
        for loc in frame.binding:
            if ZoneAddress(loc.zone_id) not in frame.effects.cz_zones:
                continue
            partner = arch.get_cz_partner(loc)
            if partner is not None and partner not in footprint:
                problems.append(
                    f"SingleZonePolicy: CZ partner {format_location(partner)} of "
                    f"{format_location(loc)} is outside the frame"
                )
        return problems
