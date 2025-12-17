import pytest

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import (
    EncodingType,
    LocationAddress,
    ZoneAddress,
)


def test_architecture_generation():
    arch_physical = generate_arch()

    assert len(arch_physical.words) == 16
    assert len(arch_physical.site_buses) == 9
    assert len(arch_physical.word_buses) == 4
    assert arch_physical.encoding is EncodingType.BIT32


def test_get_zone_index():
    arch_physical = generate_arch()

    loc_addr = LocationAddress(word_id=0, site_id=0)
    zone_id = ZoneAddress(0)
    index = arch_physical.get_zone_index(loc_addr, zone_id)
    assert index == 0

    loc_addr = LocationAddress(word_id=0, site_id=1)
    zone_id = ZoneAddress(0)
    index = arch_physical.get_zone_index(loc_addr, zone_id)
    assert index == 1

    loc_addr = LocationAddress(word_id=1, site_id=0)
    zone_id = ZoneAddress(0)
    index = arch_physical.get_zone_index(loc_addr, zone_id)
    assert index == 10


def test_logical_architecture():
    assert logical.get_arch_spec() == generate_arch(hypercube_dims=1, word_size_y=5)


def plot():
    from matplotlib import pyplot as plt

    arch_physical = generate_arch()
    f, axs = plt.subplots(1, 1)

    ax = arch_physical.plot(
        show_words=(0, 1), show_site_bus=tuple(range(4)), show_word_bus=(0,), ax=axs
    )

    ax.set_aspect(0.25)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - 2, xmax + 2)
    ax.set_ylim(ymin - 2, ymax + 2)

    f, axs = plt.subplots(2, 2, figsize=(10, 8))

    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(0,), ax=axs[0, 0])
    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(1,), ax=axs[0, 1])
    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(2,), ax=axs[1, 0])
    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(3,), ax=axs[1, 1])

    plt.show()


def invalid_locations():
    arch_spec = logical.get_arch_spec()
    yield arch_spec, LocationAddress(16, 0), set(["Word id 16 out of range of 2"])
    yield arch_spec, LocationAddress(0, 32), set(["Site id 32 out of range of 10"])
    yield arch_spec, LocationAddress(-1, 0), set(["Word id -1 out of range of 2"])
    yield arch_spec, LocationAddress(0, -1), set(["Site id -1 out of range of 10"])


@pytest.mark.parametrize("arch_spec, location_address, message", invalid_locations())
def test_location_validation(
    arch_spec: ArchSpec, location_address: LocationAddress, message: set[str]
):
    assert message == arch_spec.validate_location(location_address)
