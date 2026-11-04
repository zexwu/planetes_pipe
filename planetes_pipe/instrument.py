"""Fixed PLANETES channel layout: telescopes, baselines and ABCD outputs."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InstrumentLayout:
    """Physical output order and the corresponding P2VM matrix axes."""

    telescopes: tuple[str, ...] = ("4", "3", "2", "1")
    baselines: tuple[str, ...] = ("43", "42", "41", "32", "31", "21")
    abcd_order: tuple[str, ...] = ("A", "C", "B", "D")

    @property
    def baseline_pairs(self) -> tuple[tuple[int, int], ...]:
        """Telescope-axis indices for each physical baseline."""
        return tuple((self.telescopes.index(b[0]), self.telescopes.index(b[1]))
                     for b in self.baselines)

    @property
    def n_telescopes(self) -> int:
        return len(self.telescopes)

    @property
    def n_baselines(self) -> int:
        return len(self.baselines)

    @property
    def n_outputs(self) -> int:
        return self.n_baselines * len(self.abcd_order)

    @property
    def n_data(self) -> int:
        return self.n_telescopes + 2 * self.n_baselines

    @property
    def real_slice(self) -> slice:
        return slice(self.n_telescopes, self.n_telescopes + self.n_baselines)

    @property
    def imag_slice(self) -> slice:
        return slice(self.real_slice.stop, self.n_data)


layout = InstrumentLayout()


