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


def test_move_to_squin_logical_no_noise_uses_clean_init():
    """With add_noise=False, InsertGates gets the clean kernel, no noise kernel."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.noise_model import generate_logical_noise_model
    from bloqade.lanes.transform import MoveToSquinLogical

    model = generate_logical_noise_model()
    arch = generate_arch_hypercube(4)

    t = MoveToSquinLogical(arch_spec=arch, noise_model=model, add_noise=False)
    assert t._get_initialize_kernel() is not None
    assert t._get_initialize_noise_kernel() is None
    assert t._get_noise_model() is None


def test_move_to_squin_logical_with_noise_uses_noisy_init_only():
    """With add_noise=True, InsertGates gets None, InsertNoise gets the noisy kernel.

    This ensures only the noisy initialization is inserted, not both clean and noisy.
    """
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.noise_model import generate_logical_noise_model
    from bloqade.lanes.transform import MoveToSquinLogical

    model = generate_logical_noise_model()
    arch = generate_arch_hypercube(4)

    t = MoveToSquinLogical(arch_spec=arch, noise_model=model, add_noise=True)
    # InsertGates should NOT get an init kernel when noise is enabled
    assert t._get_initialize_kernel() is None
    # InsertNoise should get the noisy init kernel
    assert t._get_initialize_noise_kernel() is not None
    # Noise model should be passed to InsertNoise
    assert t._get_noise_model() is not None


def test_move_to_squin_logical_init_kernels_mutually_exclusive():
    """Clean and noisy init kernels are never both active at the same time."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.noise_model import generate_logical_noise_model
    from bloqade.lanes.transform import MoveToSquinLogical

    model = generate_logical_noise_model()
    arch = generate_arch_hypercube(4)

    for add_noise in (False, True):
        t = MoveToSquinLogical(arch_spec=arch, noise_model=model, add_noise=add_noise)
        clean = t._get_initialize_kernel()
        noisy = t._get_initialize_noise_kernel()
        # Exactly one should be set, never both
        assert (clean is None) != (noisy is None), (
            f"add_noise={add_noise}: clean={clean}, noisy={noisy} — "
            "expected exactly one to be set"
        )


def test_move_to_squin_physical_no_init():
    """MoveToSquinPhysical never provides init kernels."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.noise_model import generate_simple_noise_model
    from bloqade.lanes.transform import MoveToSquinPhysical

    arch = generate_arch_hypercube(4)
    model = generate_simple_noise_model()

    t = MoveToSquinPhysical(arch_spec=arch, noise_model=model)
    assert t._get_initialize_kernel() is None
    assert t._get_initialize_noise_kernel() is None
