from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.word import Word

# Direct construction of the Gemini logical arch spec using the zone-centric
# schema. This bypasses the builder to isolate the pipeline from builder bugs
# during the zone-centric migration.
#
# Architecture: 10 words x 2 sites, 2 zones (entangling pair).
# Zone 0 = even-column words (0,2,4,6,8), Zone 1 = odd-column words (1,3,5,7,9)
# Grid: 4 x-positions (interleaved even/odd with site_spacing=10),
#        5 y-positions (rows, row_spacing=20)
# Site bus: site 0 <-> site 1 (hypercube on 2 sites)
# Word buses: diagonal hypercube topology (9 buses per zone)

_ARCH_JSON = r"""
{
  "version": "2.0",
  "words": [
    {"sites": [[0, 0], [2, 0]]},
    {"sites": [[1, 0], [3, 0]]},
    {"sites": [[0, 1], [2, 1]]},
    {"sites": [[1, 1], [3, 1]]},
    {"sites": [[0, 2], [2, 2]]},
    {"sites": [[1, 2], [3, 2]]},
    {"sites": [[0, 3], [2, 3]]},
    {"sites": [[1, 3], [3, 3]]},
    {"sites": [[0, 4], [2, 4]]},
    {"sites": [[1, 4], [3, 4]]}
  ],
  "zones": [
    {
      "grid": {
        "x_start": 0.0, "y_start": 0.0,
        "x_spacing": [10.0, 10.0, 10.0],
        "y_spacing": [20.0, 20.0, 20.0, 20.0]
      },
      "site_buses": [{"src": [0], "dst": [1]}],
      "word_buses": [
        {"src": [0, 2, 4, 6, 8], "dst": [1, 3, 5, 7, 9]},
        {"src": [0, 2, 4, 6], "dst": [3, 5, 7, 9]},
        {"src": [0, 2, 4], "dst": [5, 7, 9]},
        {"src": [0, 2], "dst": [7, 9]},
        {"src": [0], "dst": [9]},
        {"src": [8], "dst": [1]},
        {"src": [6, 8], "dst": [1, 3]},
        {"src": [4, 6, 8], "dst": [1, 3, 5]},
        {"src": [2, 4, 6, 8], "dst": [1, 3, 5, 7]}
      ],
      "words_with_site_buses": [0, 2, 4, 6, 8],
      "sites_with_word_buses": [0, 1]
    },
    {
      "grid": {
        "x_start": 10.0, "y_start": 0.0,
        "x_spacing": [10.0, 10.0, 10.0],
        "y_spacing": [20.0, 20.0, 20.0, 20.0]
      },
      "site_buses": [{"src": [0], "dst": [1]}],
      "word_buses": [
        {"src": [0, 2, 4, 6, 8], "dst": [1, 3, 5, 7, 9]},
        {"src": [0, 2, 4, 6], "dst": [3, 5, 7, 9]},
        {"src": [0, 2, 4], "dst": [5, 7, 9]},
        {"src": [0, 2], "dst": [7, 9]},
        {"src": [0], "dst": [9]},
        {"src": [8], "dst": [1]},
        {"src": [6, 8], "dst": [1, 3]},
        {"src": [4, 6, 8], "dst": [1, 3, 5]},
        {"src": [2, 4, 6, 8], "dst": [1, 3, 5, 7]}
      ],
      "words_with_site_buses": [1, 3, 5, 7, 9],
      "sites_with_word_buses": [0, 1]
    }
  ],
  "zone_buses": [],
  "entangling_zone_pairs": [[0, 1]],
  "modes": [
    {"name": "all", "zones": [0, 1], "bitstring_order": []},
    {"name": "gate", "zones": [0, 1], "bitstring_order": []}
  ]
}
"""


def get_arch_spec() -> ArchSpec:
    rust_spec = _RustArchSpec.from_json(_ARCH_JSON)
    # Wrap Rust Word objects into Python Word wrappers
    py_words = []
    for rw in rust_spec.words:
        w = Word.__new__(Word)
        w._inner = rw
        py_words.append(w)
    return ArchSpec(inner=rust_spec, words=tuple(py_words))
