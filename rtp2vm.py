"""Reusable real-time P2VM reduction for PLANETES camera frames."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def extract_spec_sparse_single(img: NDArray, ys: NDArray, xs: NDArray) -> NDArray:
    """Extract one detector frame into sparse output-channel spectra."""
    _, nx = img.shape
    n_out = len(ys)
    output = np.zeros((n_out, nx), dtype=np.float32)

    for out in prange(n_out):
        y_idx = ys[out]
        x_idx = xs[out]
        for i in range(len(y_idx)):
            x = x_idx[i]
            output[out, x] += img[y_idx[i], x]
    return output


@njit(parallel=True, cache=True, boundscheck=False)
def twopx_interp_single(spec: NDArray, wn_map: NDArray, wn_grid: NDArray) -> NDArray:
    """Resample extracted spectra from detector pixels to a wavenumber grid."""
    n_reg, _ = spec.shape
    n_wave = len(wn_grid)
    out = np.zeros((n_reg, n_wave))

    for reg in prange(n_reg):
        x_in = wn_map[reg, ::-1]
        y_in = spec[reg, ::-1]
        out[reg, :] = np.interp(wn_grid, x_in, y_in)

    return out


@dataclass(slots=True)
class P2VMReductionResult:
    """Reduced visibility products for one detector frame."""

    visamp: NDArray
    flux: NDArray
    spec_aligned: NDArray


@dataclass(slots=True)
class P2VMReducer:
    """Load calibration products once and reduce individual camera frames."""

    reduced_dir: Path = Path("planetes_pipe/reduced")
    profile_ys: NDArray = field(init=False, repr=False)
    profile_xs: NDArray = field(init=False, repr=False)
    flat_map: NDArray = field(init=False, repr=False)
    dark_map: NDArray | None = field(init=False, repr=False)
    profile_x_positions: NDArray | None = field(init=False, repr=False)
    profile_y_positions: NDArray | None = field(init=False, repr=False)
    p2vm: NDArray = field(init=False, repr=False)
    wl_grid: NDArray = field(init=False, repr=False)
    bsl_to_reg: dict[int, list[int]] = field(init=False, repr=False)
    bsl_to_tel: NDArray = field(init=False, repr=False)
    wave_map: NDArray = field(init=False, repr=False)
    wn_map: NDArray = field(init=False, repr=False)
    wn_grid: NDArray = field(init=False, repr=False)
    spec_flat: NDArray = field(init=False, repr=False)
    spec_flat_aligned: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reduced_dir = Path(self.reduced_dir).expanduser().resolve()

        with np.load(self.reduced_dir / "flat.npz") as d:
            self.profile_ys = np.array(d["profile_ys"])
            self.profile_xs = np.array(d["profile_xs"])
            self.flat_map = d["flat_map"].astype(np.float32)
            self.dark_map = d["dark_map"].astype(np.float32) if "dark_map" in d else None
            self.profile_x_positions = d["xs"] if "xs" in d else None
            self.profile_y_positions = d["ys"] if "ys" in d else None

        with np.load(self.reduced_dir / "p2vm.npz", allow_pickle=True) as d:
            self.p2vm = d["p2vm"]
            self.wl_grid = d["wl_grid"]
            bsl_to_reg = d["bsl_to_reg"][()]
            bsl_to_tel = d["bsl_to_tel"][()]

        self.bsl_to_reg = {int(k): list(v) for k, v in bsl_to_reg.items()}
        baseline_ids = sorted(int(k) for k in bsl_to_tel)
        self.bsl_to_tel = np.array([list(bsl_to_tel[i]) for i in baseline_ids])

        with np.load(self.reduced_dir / "wave.npz") as d:
            self.wave_map = d["wave_map"]

        self.wn_map = 1.0 / self.wave_map
        self.wn_grid = 1.0 / self.wl_grid

        spec_flat = extract_spec_sparse_single(
            self.flat_map, self.profile_ys, self.profile_xs
        )
        self.spec_flat = np.clip(spec_flat, a_min=1e-8, a_max=None)
        self.spec_flat_aligned = twopx_interp_single(
            1.0 / self.spec_flat, self.wn_map, self.wn_grid
        )

    @property
    def n_baselines(self) -> int:
        """Number of calibrated baselines."""
        return self.bsl_to_tel.shape[0]

    @property
    def n_outputs(self) -> int:
        """Number of detector output regions."""
        return len(self.profile_ys)

    def reduce(self, img: NDArray, subtract_dark: bool = False) -> P2VMReductionResult:
        """Reduce one detector frame to normalized visibility amplitudes."""
        frame = np.asarray(img, dtype=np.float32)
        if subtract_dark and self.dark_map is not None:
            frame = frame - self.dark_map

        spec = extract_spec_sparse_single(frame, self.profile_ys, self.profile_xs)
        spec /= self.spec_flat

        spec_aligned = twopx_interp_single(spec, self.wn_map, self.wn_grid)
        spec_aligned /= self.spec_flat_aligned

        p2vmred = np.einsum("dow,ow->dw", self.p2vm, spec_aligned)
        flux = p2vmred[0:4]
        real = p2vmred[4:10]
        imag = p2vmred[10:16]
        visamp = np.sqrt(real**2 + imag**2)

        f1f2 = (flux[self.bsl_to_tel[:, 0]] * flux[self.bsl_to_tel[:, 1]]) ** 0.5
        visamp /= np.clip(f1f2, a_min=1e-8, a_max=None)

        return P2VMReductionResult(visamp=visamp, flux=flux, spec_aligned=spec_aligned)
