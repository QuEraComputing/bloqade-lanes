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


def test_generate_logical_noise_model():
    """generate_logical_noise_model returns both clean and noisy init kernels."""
    from bloqade.lanes.noise_model import generate_logical_noise_model

    model = generate_logical_noise_model()
    clean, noisy = model.get_logical_initialize()
    assert clean is not None
    assert noisy is not None


def test_generate_simple_noise_model_has_no_init():
    """generate_simple_noise_model returns a physical model without init kernels."""
    from bloqade.lanes.noise_model import generate_simple_noise_model

    model = generate_simple_noise_model()
    # SimpleNoiseModel has get_logical_initialize returning (None, None) from ABC default
    clean, noisy = model.get_logical_initialize()
    assert clean is None
    assert noisy is None


def test_move_to_squin_logical_resolves_init():
    """MoveToSquinLogical resolves clean/noisy kernels from noise model."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.noise_model import generate_logical_noise_model
    from bloqade.lanes.transform import MoveToSquinLogical

    model = generate_logical_noise_model()
    arch = generate_arch_hypercube(4)

    t = MoveToSquinLogical(arch_spec=arch, noise_model=model, add_noise=False)
    assert t._get_initialize_kernel() is not None
    assert t._get_initialize_noise_kernel() is None

    t = MoveToSquinLogical(arch_spec=arch, noise_model=model, add_noise=True)
    assert t._get_initialize_kernel() is not None
    assert t._get_initialize_noise_kernel() is not None


def test_move_to_squin_backwards_compat():
    """MoveToSquin backwards compat: explicit param takes priority."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.arch.gemini.logical import steane7_initialize
    from bloqade.lanes.transform import MoveToSquin

    arch = generate_arch_hypercube(4)

    # No noise model: returns None
    t = MoveToSquin(arch_spec=arch)
    assert t._get_initialize_kernel() is None

    # Explicit param used directly
    t = MoveToSquin(arch_spec=arch, logical_initialization=steane7_initialize)
    assert t._get_initialize_kernel() is steane7_initialize
