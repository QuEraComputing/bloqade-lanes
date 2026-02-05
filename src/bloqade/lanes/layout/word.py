from dataclasses import dataclass
from typing import Any, Generic, TypeVar

SiteType = TypeVar("SiteType", bound=tuple[float, float] | tuple[int, int])


@dataclass(frozen=True)
class Word(Generic[SiteType]):
    # note that the `SiteType` is really just here for visualization purposes
    # you can simply ignore the site in general
    sites: tuple[SiteType, ...]
    """Geometric layout of the word, consisting of one or more coordinates per site"""
    has_cz: tuple[int, ...] | None = None
    """defines which sites in the word have a controlled-Z (CZ) interaction, e.g. has_cz[i] = j means site i has a CZ with site j"""

    def __post_init__(self):
        assert len(self.sites) == len(self.has_cz) if self.has_cz is not None else True

    def __getitem__(self, index: int):
        return WordSite(
            word=self,
            site_index=index,
            cz_pair=self.has_cz[index] if self.has_cz is not None else None,
        )

    def site_position(self, site_index: int) -> tuple[float, float]:
        site = self.sites[site_index]
        return site

    def all_positions(self):
        for site_index in range(len(self.sites)):
            yield self.site_position(site_index)

    def plot(self, ax=None, **scatter_kwargs):
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

        if ax is None:
            ax = plt.gca()
        x_positions, y_positions = zip(*self.all_positions())
        ax.scatter(x_positions, y_positions, **scatter_kwargs)
        return ax


WordType = TypeVar("WordType", bound=Word[Any])


@dataclass(frozen=True)
class WordSite(Generic[WordType]):
    word: WordType
    site_index: int
    cz_pair: int | None = None

    def positions(self):
        yield self.word.site_position(self.site_index)
