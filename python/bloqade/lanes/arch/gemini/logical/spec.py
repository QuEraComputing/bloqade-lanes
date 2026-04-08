from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.word import Word

# Direct construction of the Gemini logical arch spec using the zone-centric
# schema. Bypasses the builder to isolate the pipeline during migration.
#
# Architecture: 5 words x 2 sites, 2 zones (entangling pair).
#
# Zone 0: x=[0.0, 10.0], y=[0.0, 10.0, 20.0, 30.0, 40.0]
# Zone 1: x=[2.0, 12.0], y=[0.0, 10.0, 20.0, 30.0, 40.0]  (zone 0 shifted +2.0 in x)
#
# Words (each row = 1 word, 2 sites = the 2 x-positions):
#   Word i: [(0, i), (1, i)]  for i in 0..4
#
# Site buses: zone 1 only — site 0 <-> site 1
# Word buses: none (no intra-zone row movement)
# Zone buses: 8 buses connecting zone 0 word i -> zone 1 word (i+shift)
#   for shifts +1..+4 and -1..-4
#
# entangling_zone_pairs: [(0, 1)]

_ARCH_JSON = r"""
{
  "version": "2.0",
  "words": [
    {"sites": [[0, 0], [1, 0]]},
    {"sites": [[0, 1], [1, 1]]},
    {"sites": [[0, 2], [1, 2]]},
    {"sites": [[0, 3], [1, 3]]},
    {"sites": [[0, 4], [1, 4]]}
  ],
  "zones": [
    {
      "grid": {
        "x_start": 0.0, "y_start": 0.0,
        "x_spacing": [10.0],
        "y_spacing": [10.0, 10.0, 10.0, 10.0]
      },
      "site_buses": [],
      "word_buses": [],
      "words_with_site_buses": [],
      "sites_with_word_buses": []
    },
    {
      "grid": {
        "x_start": 2.0, "y_start": 0.0,
        "x_spacing": [10.0],
        "y_spacing": [10.0, 10.0, 10.0, 10.0]
      },
      "site_buses": [{"src": [0], "dst": [1]}],
      "word_buses": [],
      "words_with_site_buses": [0, 1, 2, 3, 4],
      "sites_with_word_buses": []
    }
  ],
  "zone_buses": [
    {"src": [{"zone_id": 0, "word_id": 0}, {"zone_id": 0, "word_id": 1}, {"zone_id": 0, "word_id": 2}, {"zone_id": 0, "word_id": 3}],
     "dst": [{"zone_id": 1, "word_id": 1}, {"zone_id": 1, "word_id": 2}, {"zone_id": 1, "word_id": 3}, {"zone_id": 1, "word_id": 4}]},
    {"src": [{"zone_id": 0, "word_id": 0}, {"zone_id": 0, "word_id": 1}, {"zone_id": 0, "word_id": 2}],
     "dst": [{"zone_id": 1, "word_id": 2}, {"zone_id": 1, "word_id": 3}, {"zone_id": 1, "word_id": 4}]},
    {"src": [{"zone_id": 0, "word_id": 0}, {"zone_id": 0, "word_id": 1}],
     "dst": [{"zone_id": 1, "word_id": 3}, {"zone_id": 1, "word_id": 4}]},
    {"src": [{"zone_id": 0, "word_id": 0}],
     "dst": [{"zone_id": 1, "word_id": 4}]},
    {"src": [{"zone_id": 0, "word_id": 1}, {"zone_id": 0, "word_id": 2}, {"zone_id": 0, "word_id": 3}, {"zone_id": 0, "word_id": 4}],
     "dst": [{"zone_id": 1, "word_id": 0}, {"zone_id": 1, "word_id": 1}, {"zone_id": 1, "word_id": 2}, {"zone_id": 1, "word_id": 3}]},
    {"src": [{"zone_id": 0, "word_id": 2}, {"zone_id": 0, "word_id": 3}, {"zone_id": 0, "word_id": 4}],
     "dst": [{"zone_id": 1, "word_id": 0}, {"zone_id": 1, "word_id": 1}, {"zone_id": 1, "word_id": 2}]},
    {"src": [{"zone_id": 0, "word_id": 3}, {"zone_id": 0, "word_id": 4}],
     "dst": [{"zone_id": 1, "word_id": 0}, {"zone_id": 1, "word_id": 1}]},
    {"src": [{"zone_id": 0, "word_id": 4}],
     "dst": [{"zone_id": 1, "word_id": 0}]}
  ],
  "entangling_zone_pairs": [[0, 1]],
  "modes": [
    {"name": "all", "zones": [0, 1], "bitstring_order": []},
    {"name": "gate", "zones": [0, 1], "bitstring_order": []}
  ]
}
"""


def get_arch_spec() -> ArchSpec:
    rust_spec = _RustArchSpec.from_json(_ARCH_JSON)
    py_words = []
    for rw in rust_spec.words:
        w = Word.__new__(Word)
        w._inner = rw
        py_words.append(w)
    return ArchSpec(inner=rust_spec, words=tuple(py_words))
