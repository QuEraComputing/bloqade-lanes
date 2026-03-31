"""Tests for zone definition data model."""

import pytest

from bloqade.lanes.arch.zone import ArchBlueprint, DeviceLayout, ZoneSpec


class TestZoneSpec:
    def test_construction(self) -> None:
        spec = ZoneSpec(num_words=4, entangling=True, measurement=True)
        assert spec.num_words == 4
        assert spec.entangling is True
        assert spec.measurement is True

    def test_defaults(self) -> None:
        spec = ZoneSpec(num_words=2)
        assert spec.entangling is False
        assert spec.measurement is True

    def test_frozen(self) -> None:
        spec = ZoneSpec(num_words=4)
        with pytest.raises(AttributeError):
            spec.num_words = 8  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ZoneSpec(num_words=4, entangling=True)
        b = ZoneSpec(num_words=4, entangling=True)
        c = ZoneSpec(num_words=4, entangling=False)
        assert a == b
        assert a != c

    def test_num_words_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_words must be >= 1"):
            ZoneSpec(num_words=0)

    def test_num_words_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_words must be >= 1"):
            ZoneSpec(num_words=-1)


class TestDeviceLayout:
    def test_defaults(self) -> None:
        layout = DeviceLayout()
        assert layout.word_size_y == 5
        assert layout.site_spacing == 10.0
        assert layout.word_spacing == 10.0
        assert layout.zone_gap == 20.0

    def test_custom_values(self) -> None:
        layout = DeviceLayout(word_size_y=3, site_spacing=5.0, zone_gap=30.0)
        assert layout.word_size_y == 3
        assert layout.site_spacing == 5.0
        assert layout.zone_gap == 30.0

    def test_frozen(self) -> None:
        layout = DeviceLayout()
        with pytest.raises(AttributeError):
            layout.word_size_y = 10  # type: ignore[misc]


class TestArchBlueprint:
    def test_valid_single_zone(self) -> None:
        bp = ArchBlueprint(zones={"proc": ZoneSpec(num_words=4, entangling=True)})
        assert bp.words_per_zone == 4
        assert bp.total_words == 4
        assert bp.zone_names == ("proc",)

    def test_valid_two_zones(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_words=4, entangling=True),
                "mem": ZoneSpec(num_words=4),
            }
        )
        assert bp.words_per_zone == 4
        assert bp.total_words == 8
        assert bp.zone_names == ("proc", "mem")

    def test_valid_three_zones(self) -> None:
        bp = ArchBlueprint(
            zones={
                "proc": ZoneSpec(num_words=4, entangling=True),
                "buffer": ZoneSpec(num_words=4),
                "mem": ZoneSpec(num_words=4),
            }
        )
        assert bp.total_words == 12
        assert bp.zone_names == ("proc", "buffer", "mem")

    def test_empty_zones_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one zone"):
            ArchBlueprint(zones={})

    def test_unequal_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="same num_words"):
            ArchBlueprint(
                zones={
                    "proc": ZoneSpec(num_words=4),
                    "mem": ZoneSpec(num_words=8),
                }
            )

    def test_zone_ordering_preserved(self) -> None:
        bp = ArchBlueprint(
            zones={
                "mem": ZoneSpec(num_words=2),
                "proc": ZoneSpec(num_words=2),
                "buffer": ZoneSpec(num_words=2),
            }
        )
        assert bp.zone_names == ("mem", "proc", "buffer")

    def test_default_layout(self) -> None:
        bp = ArchBlueprint(zones={"proc": ZoneSpec(num_words=4)})
        assert bp.layout == DeviceLayout()

    def test_custom_layout(self) -> None:
        layout = DeviceLayout(word_size_y=3, zone_gap=50.0)
        bp = ArchBlueprint(
            zones={"proc": ZoneSpec(num_words=4)},
            layout=layout,
        )
        assert bp.layout.word_size_y == 3
        assert bp.layout.zone_gap == 50.0

    def test_frozen(self) -> None:
        bp = ArchBlueprint(zones={"proc": ZoneSpec(num_words=4)})
        with pytest.raises(AttributeError):
            bp.zones = {}  # type: ignore[misc]
