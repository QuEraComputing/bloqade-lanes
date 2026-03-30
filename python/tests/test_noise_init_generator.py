from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC


class MinimalNoise(NoiseModelABC):
    """Minimal concrete subclass for testing the ABC."""

    def get_lane_noise(self, lane):  # type: ignore[override]
        ...

    def get_bus_idle_noise(self, move_type, bus_id):  # type: ignore[override]
        ...

    def get_cz_unpaired_noise(self, zone_address):  # type: ignore[override]
        ...

    def get_logical_initialize(self):  # type: ignore[override]
        return None, None


def test_noise_model_abc_get_logical_initialize_is_abstract():
    """get_logical_initialize is abstract and must be implemented."""
    import pytest

    class IncompleteNoise(NoiseModelABC):
        def get_lane_noise(self, lane):  # type: ignore[override]
            ...

        def get_bus_idle_noise(self, move_type, bus_id):  # type: ignore[override]
            ...

        def get_cz_unpaired_noise(self, zone_address):  # type: ignore[override]
            ...

    with pytest.raises(TypeError):
        IncompleteNoise()  # type: ignore[abstract]


def test_noise_model_abc_get_logical_initialize_concrete():
    """Concrete subclass implementing get_logical_initialize works."""
    model = MinimalNoise()
    clean, noisy = model.get_logical_initialize()
    assert clean is None
    assert noisy is None


def test_simple_noise_model_get_logical_initialize_default():
    """SimpleNoiseModel returns (None, None) when no init kernels are set."""
    from bloqade.lanes.noise_model import generate_simple_noise_model

    model = generate_simple_noise_model()
    clean, noisy = model.get_logical_initialize()
    assert clean is None
    assert noisy is None


def test_simple_noise_model_get_logical_initialize_with_kernels():
    """SimpleNoiseModel returns provided init kernels."""
    from bloqade.lanes.arch.gemini.logical import steane7_initialize
    from bloqade.lanes.noise_model import generate_simple_noise_model

    model = generate_simple_noise_model()
    model.logical_initialize_clean = steane7_initialize
    model.logical_initialize_noisy = None

    clean, noisy = model.get_logical_initialize()
    assert clean is steane7_initialize
    assert noisy is None


def test_move_to_squin_always_uses_clean_for_insert_gates():
    """InsertGates always gets the clean kernel regardless of add_noise."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.arch.gemini.logical import steane7_initialize
    from bloqade.lanes.noise_model import generate_simple_noise_model
    from bloqade.lanes.transform import MoveToSquin

    model = generate_simple_noise_model()
    model.logical_initialize_clean = steane7_initialize

    arch = generate_arch_hypercube(4)

    # add_noise=False: clean kernel for gates, no noise kernel
    t = MoveToSquin(arch_spec=arch, noise_model=model, add_noise=False)
    assert t._resolve_initialize_kernel() is steane7_initialize
    assert t._resolve_initialize_noise_kernel() is None

    # add_noise=True: still clean kernel for gates, no noise kernel (none set)
    t = MoveToSquin(arch_spec=arch, noise_model=model, add_noise=True)
    assert t._resolve_initialize_kernel() is steane7_initialize
    assert t._resolve_initialize_noise_kernel() is None


def test_move_to_squin_noise_kernel_only_with_add_noise():
    """InsertNoise gets the noisy kernel only when add_noise=True."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.arch.gemini.logical import steane7_initialize
    from bloqade.lanes.noise_model import generate_simple_noise_model
    from bloqade.lanes.transform import MoveToSquin

    model = generate_simple_noise_model()
    model.logical_initialize_clean = steane7_initialize
    model.logical_initialize_noisy = steane7_initialize  # stand-in

    arch = generate_arch_hypercube(4)

    # add_noise=False: no noise kernel
    t = MoveToSquin(arch_spec=arch, noise_model=model, add_noise=False)
    assert t._resolve_initialize_noise_kernel() is None

    # add_noise=True: noise kernel returned
    t = MoveToSquin(arch_spec=arch, noise_model=model, add_noise=True)
    assert t._resolve_initialize_noise_kernel() is steane7_initialize


def test_move_to_squin_no_init_kernels():
    """With no init kernels on noise model, returns None."""
    from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
    from bloqade.lanes.noise_model import generate_simple_noise_model
    from bloqade.lanes.transform import MoveToSquin

    model = generate_simple_noise_model()
    arch = generate_arch_hypercube(4)
    t = MoveToSquin(arch_spec=arch, noise_model=model)

    assert t._resolve_initialize_kernel() is None
    assert t._resolve_initialize_noise_kernel() is None
