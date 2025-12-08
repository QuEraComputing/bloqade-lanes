from ..impls import generate_arch


def get_arch_spec():
    return generate_arch(hypercube_dims=1, word_size_y=5)
