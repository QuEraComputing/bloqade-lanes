import subprocess
import sys
import textwrap


def test_gemini_decoding_public_imports():
    from bloqade.gemini import decoding

    assert hasattr(decoding, "GurobiDecoderWithConfidence")
    assert hasattr(decoding, "PostSelectionExperiment")
    assert hasattr(decoding, "TableDecoderWithConfidence")
    assert hasattr(decoding, "TomographyResult")
    assert hasattr(decoding, "magic_state_dist_steane")
    assert not hasattr(decoding, "DEFAULT_TARGET_BLOCH")
    assert not hasattr(decoding, "plot_decoder_curves")


def test_gemini_imports_without_table_decoder():
    """Missing optional TableDecoder is reported only when it is constructed."""

    script = """
        import bloqade.decoders

        del bloqade.decoders.TableDecoder

        import stim
        from bloqade.gemini.decoding.table_decoders import TableDecoderWithConfidence

        try:
            TableDecoderWithConfidence(stim.DetectorErrorModel(""), num_shots=0)
        except ImportError as exc:
            assert "bloqade-lanes[msd-reprod]" in str(exc)
        else:
            raise AssertionError("expected the optional-dependency error")
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
