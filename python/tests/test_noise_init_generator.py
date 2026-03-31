from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC


class MinimalNoise(NoiseModelABC):
    """Minimal concrete subclass for testing the ABC default."""

    def get_lane_noise(self, lane):  # type: ignore[override]
        ...

    def get_bus_idle_noise(self, move_type, bus_id):  # type: ignore[override]
        ...

    def get_cz_unpaired_noise(self, zone_address):  # type: ignore[override]
        ...


def test_noise_model_abc_get_logical_initialize_default():
    """Default implementation returns (None, None)."""
    model = MinimalNoise()
    clean, noisy = model.get_logical_initialize()
    assert clean is None
    assert noisy is None


def test_simple_noise_model_get_logical_initialize_default():
    """generate_simple_noise_model returns both clean and noisy init kernels."""
    from bloqade.lanes.noise_model import generate_simple_noise_model

    model = generate_simple_noise_model()
    clean, noisy = model.get_logical_initialize()
    assert clean is not None
    assert noisy is not None


def test_simple_noise_model_get_logical_initialize_with_kernels():
    """SimpleLogicalNoiseModel returns provided init kernels."""
    from bloqade.lanes.arch.gemini.logical import steane7_initialize
    from bloqade.lanes.noise_model import generate_simple_noise_model

    model = generate_simple_noise_model()
    # Override the clean kernel to verify it's returned
    model.logical_initialize_clean = steane7_initialize

    clean, noisy = model.get_logical_initialize()
    assert clean is steane7_initialize
    assert noisy is not None


def test_move_to_squin_resolves_init_from_noise_model():
    """MoveToSquin prefers noise model init when no explicit param given."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.arch.gemini.logical import steane7_initialize
    from bloqade.lanes.noise_model import generate_simple_noise_model
    from bloqade.lanes.transform import MoveToSquin

    model = generate_simple_noise_model()
    model.logical_initialize_clean = steane7_initialize

    arch = generate_arch_hypercube(4)
    t = MoveToSquin(arch_spec=arch, noise_model=model)

    assert t._resolve_initialize_kernel() is steane7_initialize
    assert t._resolve_initialize_noise_kernel() is not None


def test_move_to_squin_explicit_param_takes_priority():
    """Explicit logical_initialization param takes priority over noise model."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.arch.gemini.logical import steane7_initialize
    from bloqade.lanes.noise_model import generate_simple_noise_model
    from bloqade.lanes.transform import MoveToSquin

    model = generate_simple_noise_model()

    arch = generate_arch_hypercube(4)
    t = MoveToSquin(
        arch_spec=arch,
        logical_initialization=steane7_initialize,
        noise_model=model,
    )

    # Explicit param takes priority over noise model's clean kernel
    assert t._resolve_initialize_kernel() is steane7_initialize


def test_move_to_squin_no_noise_model_returns_none():
    """Without noise model or explicit param, returns None."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.transform import MoveToSquin

    arch = generate_arch_hypercube(4)
    t = MoveToSquin(arch_spec=arch)

    assert t._resolve_initialize_kernel() is None
    assert t._resolve_initialize_noise_kernel() is None
