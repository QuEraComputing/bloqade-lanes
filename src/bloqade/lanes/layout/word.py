from dataclasses import dataclass

from bloqade.geometry.dialects import grid

from .encoding import LocationAddress


@dataclass(frozen=True)
class Word:
    # note that the `SiteType` is really just here for visualization purposes
    # you can simply ignore the site in general
    positions: grid.Grid
    site_indices: tuple[tuple[int, int], ...]
    """Geometric layout of the word, consisting of one or more coordinates per site"""
    has_cz: tuple[LocationAddress, ...] | None = None
    """defines which sites in the word have a controlled-Z (CZ) interaction, e.g. has_cz[i] = j means site i has a CZ with site j"""

    def __post_init__(self):
        if len(self.positions.positions) != len(self.site_indices):
            raise ValueError("Number of positions must match number of site indices")
        if self.has_cz is not None and len(self.has_cz) != len(self.site_indices):
            raise ValueError("Length of has_cz must match number of site indices")

    def __getitem__(self, index: int):
        return WordSite(
            word=self,
            site_index=index,
            cz_pair=self.has_cz[index] if self.has_cz is not None else None,
        )

    def site_position(self, site_index: int) -> tuple[float, float]:
        return self.positions.get(self.site_indices[site_index])

    def all_positions(self):
        yield from map(self.site_position, range(len(self.site_indices)))

    def plot(self, ax=None, **scatter_kwargs):
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

        if ax is None:
            ax = plt.gca()
        x_positions, y_positions = zip(*self.all_positions())
        ax.scatter(x_positions, y_positions, **scatter_kwargs)
        return ax


@dataclass(frozen=True)
class WordSite:
    word: Word
    site_index: int
    cz_pair: LocationAddress | None = None

    def position(self):
        return self.word.site_position(self.site_index)
