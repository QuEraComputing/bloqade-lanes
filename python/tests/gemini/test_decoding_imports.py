def test_gemini_decoding_public_imports():
    from bloqade.gemini import decoding

    assert hasattr(decoding, "GurobiDecoderWithConfidence")
    assert hasattr(decoding, "DecodedPostselectionResult")
    assert hasattr(decoding, "PostSelectionExperiment")
    assert hasattr(decoding, "TableDecoderWithConfidence")
    assert hasattr(decoding, "TomographyResult")
    assert hasattr(decoding, "magic_state_dist_steane")
    assert hasattr(decoding, "plot_decoder_curves")
