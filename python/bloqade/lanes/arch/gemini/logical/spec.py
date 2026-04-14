from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
from bloqade.lanes.layout.arch import ArchSpec

# Direct construction of the Gemini logical arch spec using the zone-centric
# schema. Bypasses the builder to isolate the pipeline during migration.
#
# Architecture: 20 words x 1 site, 1 gate zone.
#
# Zone 0 (gate): x=[0.0, 2.0, 10.0, 12.0], y=[0.0, 10.0, 20.0, 30.0, 40.0]
#   4 words per row, 5 rows = 20 words
#   Words are interleaved: columns at x=0,2 form one CZ pair, x=10,12 another
#
# Word layout (4 per row, row-major):
#   Row 0: W0=(0,0) W1=(1,0) W2=(2,0) W3=(3,0)
#   Row 1: W4=(0,1) W5=(1,1) W6=(2,1) W7=(3,1)
#   Row 2: W8=(0,2) W9=(1,2) W10=(2,2) W11=(3,2)
#   Row 3: W12=(0,3) W13=(1,3) W14=(2,3) W15=(3,3)
#   Row 4: W16=(0,4) W17=(1,4) W18=(2,4) W19=(3,4)
#
# Entangling pairs (adjacent x-columns at blockade distance):
#   [0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]
#
# Word buses (both column pairs merged per shift):
#   Columns 0->1 + 2->3 merged: shifts 0..+4 and -1..-4 (9 buses)
#   Columns 1->3 (x=2->12): shift 0 only (1 bus)
#   Total: 10 word buses

_ARCH_JSON = r"""
{
  "version": "2.0",
  "words": [
    {"sites": [[0, 0]]},
    {"sites": [[1, 0]]},
    {"sites": [[2, 0]]},
    {"sites": [[3, 0]]},
    {"sites": [[0, 1]]},
    {"sites": [[1, 1]]},
    {"sites": [[2, 1]]},
    {"sites": [[3, 1]]},
    {"sites": [[0, 2]]},
    {"sites": [[1, 2]]},
    {"sites": [[2, 2]]},
    {"sites": [[3, 2]]},
    {"sites": [[0, 3]]},
    {"sites": [[1, 3]]},
    {"sites": [[2, 3]]},
    {"sites": [[3, 3]]},
    {"sites": [[0, 4]]},
    {"sites": [[1, 4]]},
    {"sites": [[2, 4]]},
    {"sites": [[3, 4]]}
  ],
  "zones": [
    {
      "name": "gate",
      "grid": {
        "x_start": 0.0, "y_start": 0.0,
        "x_spacing": [2.0, 8.0, 2.0],
        "y_spacing": [10.0, 10.0, 10.0, 10.0]
      },
      "site_buses": [],
      "word_buses": [
        {"src": [0, 4, 8, 12, 16, 2, 6, 10, 14, 18], "dst": [1, 5, 9, 13, 17, 3, 7, 11, 15, 19]},
        {"src": [0, 4, 8, 12, 2, 6, 10, 14],         "dst": [5, 9, 13, 17, 7, 11, 15, 19]},
        {"src": [0, 4, 8, 2, 6, 10],                  "dst": [9, 13, 17, 11, 15, 19]},
        {"src": [0, 4, 2, 6],                          "dst": [13, 17, 15, 19]},
        {"src": [0, 2],                                "dst": [17, 19]},
        {"src": [4, 8, 12, 16, 6, 10, 14, 18],        "dst": [1, 5, 9, 13, 3, 7, 11, 15]},
        {"src": [8, 12, 16, 10, 14, 18],               "dst": [1, 5, 9, 3, 7, 11]},
        {"src": [12, 16, 14, 18],                      "dst": [1, 5, 3, 7]},
        {"src": [16, 18],                              "dst": [1, 3]},
        {"src": [1, 5, 9, 13, 17],                     "dst": [3, 7, 11, 15, 19]}
      ],
      "words_with_site_buses": [],
      "sites_with_word_buses": [0],
      "entangling_pairs": [
        [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
        [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]
      ]
    }
  ],
  "zone_buses": [],
  "modes": [
    {"name": "all", "zones": [0], "bitstring_order": []},
    {"name": "gate", "zones": [0], "bitstring_order": []}
  ]
}
"""


def get_arch_spec() -> ArchSpec:
    rust_spec = _RustArchSpec.from_json(_ARCH_JSON)
    return ArchSpec(rust_spec)
