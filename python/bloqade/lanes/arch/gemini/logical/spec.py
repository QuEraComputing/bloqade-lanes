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
    {"sites": [[0, 0]]},
    {"sites": [[0, 1]]},
    {"sites": [[0, 2]]},
    {"sites": [[0, 3]]},
    {"sites": [[0, 4]]},
    {"sites": [[1, 0]]},
    {"sites": [[1, 1]]},
    {"sites": [[1, 2]]},
    {"sites": [[1, 3]]},
    {"sites": [[1, 4]]}
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
      "site_buses": [],
      "word_buses": [
        {"src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9]}
      ],
      "words_with_site_buses": [],
      "sites_with_word_buses": [0]
    }
  ],
  "zone_buses": [
    {"src":[
        {"zone_id":0,"word_id":0},
        {"zone_id":0,"word_id":1},
        {"zone_id":0,"word_id":2},
        {"zone_id":0,"word_id":3},
        {"zone_id":0,"word_id":4}
    ],
    "dst": [
        {"zone_id":1,"word_id":0},
        {"zone_id":1,"word_id":1},
        {"zone_id":1,"word_id":2},
        {"zone_id":1,"word_id":3},
        {"zone_id":1,"word_id":4}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":0},
        {"zone_id":0,"word_id":1},
        {"zone_id":0,"word_id":2},
        {"zone_id":0,"word_id":3}
    ],
    "dst": [
        {"zone_id":1,"word_id":1},
        {"zone_id":1,"word_id":2},
        {"zone_id":1,"word_id":3},
        {"zone_id":1,"word_id":4}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":0},
        {"zone_id":0,"word_id":1},
        {"zone_id":0,"word_id":2}
    ],
    "dst": [
        {"zone_id":1,"word_id":2},
        {"zone_id":1,"word_id":3},
        {"zone_id":1,"word_id":4}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":0},
        {"zone_id":0,"word_id":1}
    ],
    "dst": [
        {"zone_id":1,"word_id":3},
        {"zone_id":1,"word_id":4}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":0}
    ],
    "dst": [
        {"zone_id":1,"word_id":4}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":4}
    ],
    "dst": [
        {"zone_id":1,"word_id":0}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":3},
        {"zone_id":0,"word_id":4}
    ],
    "dst": [
        {"zone_id":1,"word_id":0},
        {"zone_id":1,"word_id":1}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":2},
        {"zone_id":0,"word_id":3},
        {"zone_id":0,"word_id":4}
    ],
    "dst": [
        {"zone_id":1,"word_id":0},
        {"zone_id":1,"word_id":1},
        {"zone_id":1,"word_id":2}
    ]
    },
    {"src":[
        {"zone_id":0,"word_id":1},
        {"zone_id":0,"word_id":2},
        {"zone_id":0,"word_id":3},
        {"zone_id":0,"word_id":4}
    ],
    "dst": [
        {"zone_id":1,"word_id":0},
        {"zone_id":1,"word_id":1},
        {"zone_id":1,"word_id":2},
        {"zone_id":1,"word_id":3}
    ]
    }
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
    # Wrap Rust Word objects into Python Word wrappers
    py_words = []
    for rw in rust_spec.words:
        w = Word.__new__(Word)
        w._inner = rw
        py_words.append(w)
    return ArchSpec(inner=rust_spec, words=tuple(py_words))
