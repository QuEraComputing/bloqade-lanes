from ..impls import generate_arch_hypercube


def get_arch_spec():
    return generate_arch_hypercube(hypercube_dims=1, word_size_y=5)
