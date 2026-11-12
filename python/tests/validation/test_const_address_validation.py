from bloqade.lanes.dialects.arch import new_at
from bloqade.lanes.validation.const_address import ConstAddressValidation

from ._steane_kernel import kernel as steane_kernel


def test_non_const_arg_reported():
    """A new_at whose zone_id is a kernel argument is not const-foldable."""

    @steane_kernel
    def kernel(z: int):
        q = new_at(z, 0, 0)  # noqa: F841

    _, errors = ConstAddressValidation().run(kernel)
    assert len(errors) == 1
    assert "address argument 'zone_id' is not a compile-time constant" in str(errors[0])


def test_every_non_const_arg_reported():
    """All three args are checked, so one pass reports all three failures."""

    @steane_kernel
    def kernel(z: int, w: int, s: int):
        q = new_at(z, w, s)  # noqa: F841

    _, errors = ConstAddressValidation().run(kernel)
    assert len(errors) == 3
    reported = str(errors)
    for arg_name in ("zone_id", "word_id", "site_id"):
        assert f"address argument '{arg_name}'" in reported


def test_const_args_no_diagnostics():
    """Literal addresses const-fold and produce no errors.

    Out-of-range is deliberately not this pass's business -- word 10_000 does
    not exist in any bundled arch, and the range check that says so needs an
    ArchSpec this pass does not have.
    """

    @steane_kernel
    def kernel():
        q = new_at(0, 10_000, 0)  # noqa: F841

    _, errors = ConstAddressValidation().run(kernel)
    assert errors == []
