import subprocess
import sys
import textwrap


def test_logical_device_import_does_not_import_tsim():
    code = textwrap.dedent("""
        import sys

        from bloqade.gemini import logical as gemini_logical
        from bloqade.gemini import GeminiLogicalDevice

        assert gemini_logical is not None
        assert GeminiLogicalDevice.__name__ == "GeminiLogicalDevice"

        loaded_tsim = sorted(
            name
            for name in sys.modules
            if name == "tsim"
            or name.startswith("tsim.")
            or name == "bloqade.tsim"
            or name.startswith("bloqade.tsim.")
        )
        if loaded_tsim:
            raise SystemExit(f"unexpected tsim imports: {loaded_tsim}")
        """)

    subprocess.run([sys.executable, "-c", code], check=True)


def test_logical_simulator_import_does_not_import_tsim():
    code = textwrap.dedent("""
        import sys

        from bloqade.gemini import GeminiLogicalSimulator

        assert GeminiLogicalSimulator.__name__ == "GeminiLogicalSimulator"

        loaded_tsim = sorted(
            name
            for name in sys.modules
            if name == "tsim"
            or name.startswith("tsim.")
            or name == "bloqade.tsim"
            or name.startswith("bloqade.tsim.")
        )
        if loaded_tsim:
            raise SystemExit(f"unexpected tsim imports: {loaded_tsim}")
        """)

    subprocess.run([sys.executable, "-c", code], check=True)


def test_tsim_backend_import_is_lazy():
    code = textwrap.dedent("""
        import sys

        from bloqade.gemini import (
            AbstractSimulatorBackend,
            BackendSample,
            CliffTSimulatorBackend,
            TsimSimulatorBackend,
        )

        assert AbstractSimulatorBackend.__name__ == "AbstractSimulatorBackend"
        assert BackendSample.__name__ == "BackendSample"
        assert CliffTSimulatorBackend.__name__ == "CliffTSimulatorBackend"
        assert TsimSimulatorBackend.__name__ == "TsimSimulatorBackend"
        assert not any(
            name == "tsim"
            or name.startswith("tsim.")
            or name == "bloqade.tsim"
            or name.startswith("bloqade.tsim.")
            for name in sys.modules
        )
        """)

    subprocess.run([sys.executable, "-c", code], check=True)
