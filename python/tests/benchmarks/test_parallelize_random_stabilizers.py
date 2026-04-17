from __future__ import annotations

from benchmarks.utils.parallelize_random_stabilizers import parallelize_source


def test_parallelize_source_combines_adjacent_disjoint_cz_only():
    source = "\n".join(
        [
            '"""demo"""',
            "",
            "from bloqade import squin",
            "",
            "@squin.kernel(typeinfer=True, fold=True)",
            "def demo():",
            "    q = squin.qalloc(10)",
            "    squin.cz(q[1], q[3])",
            "    squin.cz(q[1], q[2])",
            "    squin.cz(q[1], q[5])",
            "    squin.cz(q[1], q[7])",
            "    squin.cz(q[2], q[4])",
            "    squin.cz(q[2], q[7])",
            "",
        ]
    )

    out = parallelize_source(source)

    assert (
        "squin.broadcast.cz(ilist.IList([q[1], q[2]]), ilist.IList([q[7], q[4]]))"
        in out
    )
    assert "squin.cz(q[1], q[3])" in out
    assert "squin.cz(q[1], q[2])" in out
    assert "squin.cz(q[1], q[5])" in out
    assert "squin.cz(q[2], q[7])" in out


def test_parallelize_source_adds_ilist_import_when_needed():
    source = "\n".join(
        [
            '"""demo"""',
            "",
            "from bloqade import squin",
            "",
            "@squin.kernel(typeinfer=True, fold=True)",
            "def demo():",
            "    q = squin.qalloc(4)",
            "    squin.cz(q[0], q[1])",
            "    squin.cz(q[2], q[3])",
            "",
        ]
    )

    out = parallelize_source(source)

    assert "from kirin.dialects import ilist" in out
    assert "squin.broadcast.cz(" in out
