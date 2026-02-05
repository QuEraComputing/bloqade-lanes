from bloqade.lanes import layout
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout.word import Word


def test_get_blockaded_location_with_pair():
    """Test get_blockaded_location returns the correct paired location."""
    arch_spec = logical.get_arch_spec()

    # location (0, 0) should pair with (0, 5)
    location = layout.LocationAddress(0, 0)
    blockaded = arch_spec.get_blockaded_location(location)

    assert blockaded is not None
    assert blockaded == layout.LocationAddress(0, 5)

    # test reverse
    location2 = layout.LocationAddress(0, 5)
    blockaded2 = arch_spec.get_blockaded_location(location2)

    assert blockaded2 is not None
    assert blockaded2 == layout.LocationAddress(0, 0)


def test_get_blockaded_location_without_pair():
    """Test get_blockaded_location returns None for locations without pairs."""

    # archspec wno sites have CZ pairs
    word = Word(
        sites=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), has_cz=None  # No CZ pairs defined
    )

    arch_spec = layout.ArchSpec(
        (word,),
        ((0,),),
        (0,),
        frozenset([0]),
        frozenset(),
        frozenset(),
        (),
        (),
        (),
    )

    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 0)) is None
    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 1)) is None
    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 2)) is None


def test_get_blockaded_location_multiple_words():
    """Test get_blockaded_location works across different words."""

    # Create ArchSpec with 4 words, each word having 4 sites: site 0 <-> site 2, site 1 <-> site 3
    words = tuple(
        Word(
            sites=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)), has_cz=(2, 3, 0, 1)
        )
        for _ in range(4)
    )

    arch_spec = layout.ArchSpec(
        words,
        (tuple(range(4)),),  # All 4 words in zone 0
        (0,),
        frozenset([0]),
        frozenset(),
        frozenset(),
        (),
        (),
        tuple(frozenset([i]) for i in range(4)),
    )

    # Test word 0: site 0 should pair with site 2
    blockaded = arch_spec.get_blockaded_location(layout.LocationAddress(0, 0))
    assert blockaded is not None
    assert blockaded == layout.LocationAddress(0, 2)

    # Test word 0: site 2 should pair with site 0
    blockaded2 = arch_spec.get_blockaded_location(layout.LocationAddress(0, 2))
    assert blockaded2 is not None
    assert blockaded2 == layout.LocationAddress(0, 0)

    # Test word 1: site 1 should pair with site 3
    blockaded3 = arch_spec.get_blockaded_location(layout.LocationAddress(1, 1))
    assert blockaded3 is not None
    assert blockaded3 == layout.LocationAddress(1, 3)

    # Test word 2: site 0 should pair with site 2
    blockaded4 = arch_spec.get_blockaded_location(layout.LocationAddress(2, 0))
    assert blockaded4 is not None
    assert blockaded4 == layout.LocationAddress(2, 2)

    # Test word 3: site 3 should pair with site 1
    blockaded5 = arch_spec.get_blockaded_location(layout.LocationAddress(3, 3))
    assert blockaded5 is not None
    assert blockaded5 == layout.LocationAddress(3, 1)
