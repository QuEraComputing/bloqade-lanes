"""Guard against drift between the documented ArchSpec JSON Schema and the
implemented format.

The schema at docs/src/arch/archspec-schema.json is the normative format
documentation; it previously described a pre-#398 format the Rust parser
cannot read, and nothing caught it. These tests pin the schema to reality:
every shipped and example spec must validate against it, and the schema
itself must be a valid Draft 2020-12 document.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "docs" / "src" / "arch" / "archspec-schema.json"

SPEC_PATHS = sorted(
    list((REPO_ROOT / "examples" / "arch").glob("*.json"))
    + list(
        (REPO_ROOT / "python" / "bloqade" / "lanes" / "arch" / "gemini").glob(
            "*/*.json"
        )
    )
)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_spec_files_found():
    # If the layout moves, fail loudly instead of silently testing nothing.
    assert SPEC_PATHS, "no arch spec JSON files found to validate"


@pytest.mark.parametrize("spec_path", SPEC_PATHS, ids=lambda p: p.name)
def test_spec_validates_against_schema(
    validator: Draft202012Validator, spec_path: Path
):
    spec = json.loads(spec_path.read_text())
    errors = [
        f"{'/'.join(map(str, e.path))}: {e.message}"
        for e in validator.iter_errors(spec)
    ]
    assert (
        not errors
    ), f"{spec_path} does not match the documented schema:\n" + "\n".join(errors)


def test_old_format_is_rejected(validator: Draft202012Validator):
    # The pre-#398 top-level shape (geometry/buses/entangling_zones/
    # measurement_mode_zones) is not the implemented format and must not
    # validate.
    old = {
        "version": "2.0",
        "geometry": {"sites_per_word": 2, "words": []},
        "buses": {"site_buses": [], "word_buses": []},
        "words_with_site_buses": [],
        "sites_with_word_buses": [],
        "zones": [{"words": [0]}],
        "entangling_zones": [],
        "measurement_mode_zones": [0],
    }
    assert list(validator.iter_errors(old)), "old pre-#398 format wrongly accepted"
