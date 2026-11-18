"""The entropy-tree viewer's architecture backdrop: its sites and extent."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from tests._arch_fixtures import load_arch

from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.visualize.entropy_tree._arch_sites import ArchSites


def _sites(arch_spec: ArchSpec, word_id: int, zone_id: int = 0):
    return [
        arch_spec.get_position(LocationAddress(word_id, site_id, zone_id))
        for site_id in range(len(arch_spec.words[word_id].site_indices))
    ]


class _FakeInner:
    """Stands in for the Rust spec: site positions keyed by (word, site, zone)."""

    def __init__(self, positions: dict[tuple[int, int, int], tuple[float, float]]):
        self.positions = positions

    def location_position(self, location):
        return self.positions.get(
            (location.word_id, location.site_id, location.zone_id)
        )


def _fake_arch(
    positions: dict[tuple[int, int, int], tuple[float, float]],
    *,
    num_words: int,
    sites_per_word: int,
    num_zones: int,
    paths: dict | None = None,
) -> ArchSpec:
    """An object with just the parts of ``ArchSpec`` that ``ArchSites`` reads."""
    word = SimpleNamespace(site_indices=range(sites_per_word))
    fake = SimpleNamespace(
        words=(word,) * num_words,
        zones=(None,) * num_zones,
        paths=paths or {},
        _inner=_FakeInner(positions),
    )
    return cast(ArchSpec, fake)


@pytest.fixture(scope="module")
def physical() -> ArchSpec:
    return get_arch_spec()


def test_path_bounds_cover_every_site_and_waypoint(physical: ArchSpec) -> None:
    points = [
        position
        for zone_id in range(len(physical.zones))
        for word_id in range(len(physical.words))
        for position in _sites(physical, word_id, zone_id)
    ]
    points += [point for path in physical.paths.values() for point in path]

    assert ArchSites(physical).path_bounds() == (
        min(x for x, _ in points),
        max(x for x, _ in points),
        min(y for _, y in points),
        max(y for _, y in points),
    )


def test_path_bounds_include_waypoints_outside_the_sites() -> None:
    arch = _fake_arch(
        {(0, 0, 0): (0.0, 0.0), (0, 1, 0): (2.0, 1.0)},
        num_words=1,
        sites_per_word=2,
        num_zones=1,
        paths={"lane": ((0.0, 0.0), (0.0, 5.0), (2.0, 5.0), (2.0, 1.0))},
    )

    assert ArchSites(arch).path_bounds() == (0.0, 2.0, 0.0, 5.0)


def test_path_bounds_without_sites_default_to_the_unit_box() -> None:
    empty = _fake_arch({}, num_words=0, sites_per_word=0, num_zones=0)

    assert ArchSites(empty).path_bounds() == (-1.0, 1.0, -1.0, 1.0)


def test_path_bounds_without_sites_still_cover_waypoints() -> None:
    arch = _fake_arch(
        {},
        num_words=0,
        sites_per_word=0,
        num_zones=0,
        paths={"lane": ((3.0, -4.0), (5.0, 2.0))},
    )

    assert ArchSites(arch).path_bounds() == (-1.0, 5.0, -4.0, 2.0)


def test_plot_scatters_each_requested_word_once(physical: ArchSpec) -> None:
    ax = MagicMock()
    words = [0, 3, 7]

    returned = ArchSites(physical).plot(
        ax, show_words=words, facecolors="none", linewidths=0.8
    )

    assert returned is ax
    assert ax.scatter.call_count == len(words)
    for call, word_id in zip(ax.scatter.call_args_list, words):
        xs, ys = call.args
        assert list(zip(xs, ys)) == _sites(physical, word_id)
        assert call.kwargs == {"facecolors": "none", "linewidths": 0.8}


def test_plot_draws_a_shared_word_template_once() -> None:
    """A word template shared by several zones is drawn from its first zone."""
    arch = load_arch("proc_mem_two_zone")
    assert len(arch.zones) == 2
    ax = MagicMock()

    ArchSites(arch).plot(ax, show_words=[0])

    (call,) = ax.scatter.call_args_list
    assert list(zip(*call.args)) == _sites(arch, 0, zone_id=0)


def test_plot_takes_a_word_from_the_first_zone_that_has_it() -> None:
    arch = _fake_arch(
        {(0, 0, 1): (4.0, 4.0), (0, 1, 1): (5.0, 4.0)},
        num_words=1,
        sites_per_word=2,
        num_zones=2,
    )
    ax = MagicMock()

    ArchSites(arch).plot(ax, show_words=[0])

    (call,) = ax.scatter.call_args_list
    assert list(zip(*call.args)) == [(4.0, 4.0), (5.0, 4.0)]


def test_plot_skips_words_without_sites() -> None:
    arch = _fake_arch(
        {(1, 0, 0): (1.0, 1.0)}, num_words=2, sites_per_word=1, num_zones=1
    )
    ax = MagicMock()

    ArchSites(arch).plot(ax, show_words=[0, 1])

    (call,) = ax.scatter.call_args_list
    assert list(zip(*call.args)) == [(1.0, 1.0)]


def test_plot_with_no_words_draws_nothing(physical: ArchSpec) -> None:
    ax = MagicMock()

    ArchSites(physical).plot(ax)

    ax.scatter.assert_not_called()
