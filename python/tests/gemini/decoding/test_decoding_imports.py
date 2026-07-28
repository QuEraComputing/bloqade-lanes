import builtins
import importlib

import pytest


def test_gemini_decoding_public_imports():
    from bloqade.gemini import decoding

    assert hasattr(decoding, "GurobiDecoderWithConfidence")
    assert hasattr(decoding, "PostSelectionExperiment")
    assert hasattr(decoding, "TableDecoderWithConfidence")
    assert hasattr(decoding, "TomographyResult")
    assert hasattr(decoding, "magic_state_dist_steane")
    assert not hasattr(decoding, "DEFAULT_TARGET_BLOCH")
    assert not hasattr(decoding, "plot_decoder_curves")


def test_confidence_imports_without_gurobi_decoder(monkeypatch):
    """`bloqade.gemini` (hence `bloqade.lanes`) must import even when the
    optional Gurobi decoder is unavailable; the error is deferred to use.

    Regression test for #840: `bloqade-decoders` only exports `GurobiDecoder`
    with its `mle` extra installed, and an eager top-level import broke the
    whole import chain for downstream consumers without that extra.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bloqade.decoders" and fromlist and "GurobiDecoder" in fromlist:
            raise ImportError("cannot import name 'GurobiDecoder'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    confidence = importlib.reload(
        importlib.import_module("bloqade.gemini.decoding.confidence")
    )

    # Import succeeds and the public symbol is still present...
    cls = confidence.GurobiDecoderWithConfidence
    # ...but constructing it without the optional dependency raises clearly.
    with pytest.raises(ImportError, match="bloqade-lanes\\[decoding\\]"):
        cls()


@pytest.fixture(autouse=True)
def _restore_confidence_module():
    """Reload confidence after the missing-Gurobi test so real state is back."""
    yield
    importlib.reload(importlib.import_module("bloqade.gemini.decoding.confidence"))
