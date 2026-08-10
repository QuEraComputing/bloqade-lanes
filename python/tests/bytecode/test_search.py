"""Tests for the Python surface of the search option bundles.

Covers ``SolveOptions.backwards_search`` — the flag that asks the solver to
plan ``target -> initial`` and then reverse-and-invert the resulting layers —
and its read-back on a built ``MoveSearch``, which exposes no ``options``
property of its own.
"""

from __future__ import annotations

from bloqade.lanes.bytecode._native import MoveSearch, SolveOptions


def test_backwards_search_defaults_to_false():
    assert SolveOptions().backwards_search is False


def test_backwards_search_round_trips_through_the_constructor():
    assert SolveOptions(backwards_search=True).backwards_search is True


def test_move_search_reports_the_backwards_search_flag_it_carries():
    search = MoveSearch.entropy().with_options(SolveOptions(backwards_search=True))

    assert search.backwards_search is True


def test_move_search_defaults_to_forward_solving():
    assert MoveSearch.entropy().backwards_search is False
