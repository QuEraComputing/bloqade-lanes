"""The typed result enums: ``SolveStatus``, ``Termination`` and ``Proof``.

They replaced string labels, so comparing a member with a string must fail
loudly, naming the member to use instead, rather than silently reading
``False``. All three share one binding macro, so every member is checked.
"""

from __future__ import annotations

import pytest

from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.bytecode._native import (
    MoveSearch,
    Proof,
    SearchEngine,
    SearchStrategy,
    SolveOptions,
    SolveStatus,
    TargetSolver,
    Termination,
)
from bloqade.lanes.bytecode.encoding import LocationAddress

MEMBERS = [
    (SolveStatus, "SolveStatus", ["SOLVED", "UNSOLVABLE", "BUDGET_EXCEEDED"]),
    (Termination, "Termination", ["BUDGET", "EXHAUSTED", "STOPPED"]),
    (Proof, "Proof", ["OPTIMAL", "NO_PLAN"]),
]

CASES = [
    pytest.param(getattr(enum, name), pyname, name, id=f"{pyname}.{name}")
    for enum, pyname, names in MEMBERS
    for name in names
]


@pytest.mark.parametrize("member, pyname, name", CASES)
def test_string_comparison_raises_and_names_the_member(member, pyname, name):
    for label in (name, name.lower(), "anything"):
        with pytest.raises(TypeError, match=rf"{pyname}\.{name}\b"):
            _ = member == label
        with pytest.raises(TypeError, match=rf"{pyname}\.{name}\b"):
            _ = member != label


@pytest.mark.parametrize("member, pyname, name", CASES)
def test_members_compare_by_identity_and_are_unordered(member, pyname, name):
    enum = type(member)
    others = [getattr(enum, n) for _, p, names in MEMBERS if p == pyname for n in names]
    for other in others:
        assert (member == other) == (other.name == name)
        assert (member != other) == (other.name != name)
    assert member.name == name
    assert hash(member) == hash(getattr(enum, name))
    with pytest.raises(TypeError, match="not ordered"):
        _ = member < getattr(enum, name)  # type: ignore[operator]


RETIRED_LABELS = [
    (SolveStatus.SOLVED, "solved"),
    (SolveStatus.UNSOLVABLE, "unsolvable"),
    (SolveStatus.BUDGET_EXCEEDED, "budget_exceeded"),
    (Termination.BUDGET, "budget"),
    (Termination.EXHAUSTED, "exhausted"),
    (Termination.STOPPED, "stopped"),
]


@pytest.mark.parametrize(
    "member, label", [pytest.param(m, lab, id=lab) for m, lab in RETIRED_LABELS]
)
def test_hashed_lookup_against_the_retired_label_raises(member, label):
    """Set and dict lookups hash before they compare, so each member hashes
    like the label it replaced: the stale lookup reaches ``==`` and raises,
    rather than silently missing."""
    with pytest.raises(TypeError):
        _ = member in {label}
    with pytest.raises(TypeError):
        _ = {label: 1}.get(member)


def test_push_rotate_proves_no_plan():
    """A target held by an immovable atom has no plan, and Push and Rotate,
    unlike a search, proves it: ``UNSOLVABLE`` with ``Proof.NO_PLAN``."""
    arch = physical.get_arch_spec()
    engine = SearchEngine.from_arch_spec(arch._inner)
    start = LocationAddress(0, 0, 0)
    held = LocationAddress(1, 0, 0)
    search = MoveSearch.ids().with_options(
        SolveOptions(strategy=SearchStrategy.PUSH_ROTATE)
    )
    result = TargetSolver(engine, search).solve(
        {0: start._inner}, {0: held._inner}, [held._inner], None
    )
    assert result.status == SolveStatus.UNSOLVABLE
    assert result.proof == Proof.NO_PLAN
