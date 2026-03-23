from bloqade.gemini.arch.impls import generate_arch_hypercube


def get_arch_spec():
    return generate_arch_hypercube(hypercube_dims=4, word_size_y=5)
