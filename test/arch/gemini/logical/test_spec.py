from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.layout.encoding import (
    EncodingType,
)


def test_architecture_generation():
    arch_physical = generate_arch()

    assert len(arch_physical.words) == 16
    assert len(arch_physical.site_buses) == 9
    assert len(arch_physical.word_buses) == 4
    assert arch_physical.encoding is EncodingType.BIT32


def test_logical_architecture():
    assert logical.get_arch_spec() == generate_arch(hypercube_dims=1, word_size_y=5)


def plot():
    from matplotlib import pyplot as plt

    arch_physical = generate_arch()
    f, axs = plt.subplots(1, 1)

    ax = arch_physical.plot(
        show_words=(0, 1), show_site_bus=tuple(range(4)), show_word_bus=(0,), ax=axs
    )

    ax.set_aspect(0.25)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - 2, xmax + 2)
    ax.set_ylim(ymin - 2, ymax + 2)

    f, axs = plt.subplots(2, 2, figsize=(10, 8))

    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(0,), ax=axs[0, 0])
    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(1,), ax=axs[0, 1])
    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(2,), ax=axs[1, 0])
    arch_physical.plot(show_words=tuple(range(16)), show_word_bus=(3,), ax=axs[1, 1])

    plt.show()
