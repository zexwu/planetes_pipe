"""Typed in-memory products produced by the reduction recipes."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


Array = NDArray[Any]


@dataclass(frozen=True, slots=True)
class FlatProduct:
    """Detector maps and sparse spectral traces."""

    profile_map: Array
    profile_xs: Array
    profile_ys: Array
    flat_map: Array
    dark_map: Array
    bad_map: Array
    xs: Array
    ys: Array


@dataclass(frozen=True, slots=True)
class WaveProduct:
    """Wavelength at every extracted detector pixel."""

    wave_map: Array


@dataclass(frozen=True, slots=True)
class P2VMPreprocProduct:
    """Calibration spectra resampled onto the common wavelength grid."""

    spec_tel: Array
    spec_bsl: Array
    spec_wavesc: Array
    spec_flat: Array
    tel_regs: Mapping[int, list[int]]
    bsl_regs: Mapping[int, list[int]]
    bsl_to_reg: Mapping[int, list[int]]
    bsl_to_tel: Mapping[int, tuple[int, int]]
    wl_grid: Array


@dataclass(frozen=True, slots=True)
class ObjectPreprocProduct:
    """One target spectrum on the common wavelength grid."""

    spec: Array
    spec_flat: Array
    wl_grid: Array


@dataclass(frozen=True, slots=True)
class P2VMProduct:
    """Visibility-to-pixel model and its inverse."""

    p2vm: Array
    v2pm: Array
    opd_per_baseline: Array
    gd_per_baseline: Array
    wl_grid: Array
    bsl_to_reg: Mapping[int, list[int]]
    bsl_to_tel: Mapping[int, tuple[int, int]]
    ellipse_results: Array


@dataclass(frozen=True, slots=True)
class ReducedProduct:
    """Reduced photometry and complex visibility observables."""

    fluxdata: Array
    coherent_flux: Array
    visamp: Array
    flag: Array
    p2vmred: Array
    wl_grid: Array
    gdelay: Array
    visphi: Array
    bsl_to_reg: Mapping[int, list[int]]
    bsl_to_tel: Mapping[int, tuple[int, int]]
