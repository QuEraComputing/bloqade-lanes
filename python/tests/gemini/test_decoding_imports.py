def test_gemini_decoding_public_imports():
    from bloqade.gemini import decoding

    assert hasattr(decoding, "BasisDataset")
    assert hasattr(decoding, "DecoderWorkflowConfig")
    assert hasattr(decoding, "MSDDecoderWorkflowConfig")
    assert hasattr(decoding, "GeminiDecoderTask")
    assert hasattr(decoding, "ObservableFrame")
    assert hasattr(decoding, "SyndromeLayout")
    assert hasattr(decoding, "build_msd_primitives")
    assert hasattr(decoding, "evaluate_curve")
