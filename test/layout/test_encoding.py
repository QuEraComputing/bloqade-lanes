from bloqade.lanes.layout.encoding import USE_HEX_REPR


def test_default_hex_repr():
    assert (
        USE_HEX_REPR is True
    ), "Expected USE_HEX_REPR to be True by default, please set it back to True"
