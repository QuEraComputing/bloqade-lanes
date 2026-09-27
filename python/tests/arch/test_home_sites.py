"""Home sites and home positions are decided per zone.

The word template is spec-wide, so every zone lays out every word. Word 1
below is the staging word of the gate zone's entangling pair; the storage zone
has no pairs, so there word 1 is a home position.
"""

from __future__ import annotations

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import ArchSpec as RustArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress

_GATE_AND_STORAGE_JSON = """
{
  "version": "2.0",
  "words": [{ "sites": [[0, 0]] }, { "sites": [[1, 0]] }],
  "zones": [
    {
      "name": "gate",
      "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [2.0], "y_spacing": [] },
      "site_buses": [], "word_buses": [],
      "words_with_site_buses": [], "sites_with_word_buses": [],
      "entangling_pairs": [[0, 1]]
    },
    {
      "name": "storage",
      "grid": { "x_start": 0.0, "y_start": 10.0, "x_spacing": [2.0], "y_spacing": [] },
      "site_buses": [], "word_buses": [],
      "words_with_site_buses": [], "sites_with_word_buses": [],
      "entangling_pairs": []
    }
  ],
  "zone_buses": [
    { "src": [{ "zone_id": 1, "word_id": 1 }], "dst": [{ "zone_id": 0, "word_id": 1 }] }
  ],
  "modes": [{ "name": "default", "zones": [0, 1], "bitstring_order": [] }]
}
"""


def _arch() -> ArchSpec:
    return ArchSpec(RustArchSpec.from_json_validated(_GATE_AND_STORAGE_JSON))


def test_home_sites_span_zones():
    assert _arch().home_sites == frozenset(
        {
            LocationAddress(0, 0, zone_id=0),
            LocationAddress(0, 0, zone_id=1),
            LocationAddress(1, 0, zone_id=1),
        }
    )


def test_location_at_resolves_in_every_zone():
    arch = _arch()
    # Grid (row 0, col 1) is word 1 site 0 in both zones.
    for zone_id in (0, 1):
        assert arch.location_at(zone_id, 0, 1) == LocationAddress(1, 0, zone_id=zone_id)


def test_is_home_position_is_per_zone():
    arch = _arch()
    assert arch.is_home_position(LocationAddress(1, 0, zone_id=1))
    assert not arch.is_home_position(LocationAddress(1, 0, zone_id=0))
    for addr in arch.home_sites:
        assert arch.is_home_position(addr)
