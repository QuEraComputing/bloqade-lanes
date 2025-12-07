from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.dialects import ilist, py
from matplotlib import pyplot as plt

from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch
from bloqade.lanes.arch.gemini.logical.rewrite import RewriteMoves
from bloqade.lanes.arch.gemini.logical.stmts import SiteBusMove
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    Direction,
    EncodingType,
    SiteLaneAddress,
)


def test_architecture_generation():
    arch_physical = generate_arch()

    assert len(arch_physical.words) == 16
    assert len(arch_physical.site_buses) == 9
    assert len(arch_physical.word_buses) == 4
    assert arch_physical.encoding is EncodingType.BIT32


def test_logical_architecture():
    assert logical.get_arch_spec() == generate_arch(hypercube_dims=1, word_size_y=5)


def test_logical_architecture_rewrite_site():

    test_block = ir.Block()

    test_block.stmts.append(
        move.Move(
            lanes=(
                SiteLaneAddress(Direction.FORWARD, 0, 0, 0),
                SiteLaneAddress(Direction.FORWARD, 0, 2, 0),
                SiteLaneAddress(Direction.FORWARD, 0, 4, 0),
                SiteLaneAddress(Direction.FORWARD, 0, 6, 0),
            )
        )
    )

    rewrite_rule = rewrite.Walk(RewriteMoves())

    rewrite_rule.rewrite(test_block)

    expected_block = ir.Block()
    expected_block.stmts.append(
        const_list := py.Constant(ilist.IList([True, True, True, True, False]))
    )
    expected_block.stmts.append(
        SiteBusMove(
            y_mask=const_list.result,
            word=0,
            bus_id=0,
            direction=Direction.FORWARD,
        )
    )
    assert_nodes(test_block, expected_block)


def test_logical_architecture_rewrite_site_no_lanes():

    test_block = ir.Block()

    test_block.stmts.append(move.Move(lanes=()))

    rewrite_rule = rewrite.Walk(RewriteMoves())
    result = rewrite_rule.rewrite(test_block)
    assert not result.has_done_something


def plot():
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
