import bloqade.gemini as gemini
import bloqade.lanes as lanes


def test_root_exports_user_facing_simulator_and_compile_helpers():
    assert gemini.logical is not None
    assert gemini.GeminiLogicalSimulator is not None
    assert gemini.GeminiLogicalSimulatorTask is not None
    assert gemini.Result is not None
    assert gemini.DetectorResult is not None
    assert gemini.compile_squin_to_move is not None
    assert gemini.compile_squin_to_move_and_visualize is not None
    assert gemini.compile_to_physical_squin_noise_model is not None
    assert gemini.compile_to_stim_program is not None
    assert gemini.transversal_rewrites is not None
    assert gemini.append_measurements_and_annotations is not None
    assert gemini.run_squin_kernel_validation is not None
    assert gemini.generate_simple_noise_model is not None
    assert gemini.NoiseModelABC is not None
    assert gemini.steane7_m2dets is not None
    assert gemini.steane7_m2obs is not None


def test_lanes_compatibility_aliases_still_work():
    assert gemini.GeminiLogicalSimulator is lanes.GeminiLogicalSimulator
    assert gemini.GeminiLogicalSimulatorTask is lanes.GeminiLogicalSimulatorTask
    assert gemini.Result is lanes.Result
    assert gemini.DetectorResult is lanes.DetectorResult
    assert gemini.generate_simple_noise_model is lanes.generate_simple_noise_model
    assert gemini.steane7_m2dets is lanes.steane7_m2dets
    assert gemini.steane7_m2obs is lanes.steane7_m2obs
