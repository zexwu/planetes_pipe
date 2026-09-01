from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from . import log
from .config import FrameRange, PipelineConfig
from .fits import CubeSelection, iter_cube
from .numerics import frame_mean
from .product_io import read_product, write_product
from .products import FlatProduct, ObjectPreprocProduct, P2VMPreprocProduct, WaveProduct

ABCD_ORDER = ("A", "C", "B", "D")


@dataclass(frozen=True, slots=True)
class _ExtractionModel:
    flat: FlatProduct
    wave_map: NDArray
    detector_flat: NDArray
    aligned_flat: NDArray
    wavelength_grid: NDArray


def order_abcd_regions(
    regions: Sequence[int], detector_order: Sequence[str]
) -> list[int]:
    """Return detector regions in the canonical internal A, C, B, D order."""
    region_ids = sorted(int(region) for region in regions)
    labels = [str(label).upper() for label in detector_order]

    if len(region_ids) != 4 or len(set(region_ids)) != 4:
        raise ValueError(f"Expected four unique ABCD regions, got {region_ids}")
    if len(labels) != 4 or set(labels) != set(ABCD_ORDER):
        raise ValueError(
            f"abcd_order must contain each of {ABCD_ORDER} exactly once, got {labels}"
        )

    region_by_label = dict(zip(labels, region_ids, strict=True))
    return [region_by_label[label] for label in ABCD_ORDER]


def run_preproc_p2vm(config: PipelineConfig) -> P2VMPreprocProduct:
    """Extract and resample the P2VM calibration cubes."""
    log.info("--- Step: PREPROC P2VM ---")
    model = _extraction_model(config)
    calibration = config.calibration
    instrument = config.instrument

    log.info("Preprocessing P2VM and FLAT files...")
    spec_bsl = np.asarray(
        [
            _extract_cube(
                config,
                calibration.p2vm[name],
                calibration.frames,
                model.flat.dark_map,
                model,
            )
            for name in instrument.baselines
        ]
    )
    spec_tel = np.asarray(
        [
            _extract_cube(
                config,
                calibration.flats[name],
                calibration.frames,
                model.flat.dark_map,
                model,
            )
            for name in instrument.telescopes
        ]
    )

    spec_wavesc = np.empty(
        (instrument.n_outputs, 0, len(model.wavelength_grid)), dtype=np.float64
    )
    if phase_reference := calibration.phase_reference:
        phase_dark = model.flat.dark_map
        if phase_reference.dark is not None:
            phase_dark = _mean_cube(config, phase_reference.dark, phase_reference.frames)
        spec_wavesc = _extract_cube(
            config,
            phase_reference.data,
            phase_reference.frames,
            phase_dark,
            model,
        )

    tel_regs = {}
    for telescope, flux in enumerate(spec_tel):
        mean_flux = flux.mean(axis=(1, 2))
        active = np.argsort(mean_flux)[-instrument.n_outputs // 2 :]
        tel_regs[telescope] = np.sort(active).tolist()

    bsl_regs = {}
    for baseline, flux in enumerate(spec_bsl):
        std_flux = flux.std(axis=1).mean(axis=-1)
        active = np.argsort(std_flux)[-4:]
        bsl_regs[baseline] = np.sort(active).tolist()

    bsl_to_reg = {}
    bsl_to_tel = {}
    detector_order = [label.value for label in instrument.abcd_order]
    for telescope_1 in range(instrument.n_telescopes):
        for telescope_2 in range(instrument.n_telescopes):
            if telescope_1 >= telescope_2:
                continue
            baseline_name = (
                instrument.telescopes[telescope_1]
                + instrument.telescopes[telescope_2]
            )
            baseline = instrument.baselines.index(baseline_name)
            from_telescopes = set(tel_regs[telescope_1]) & set(tel_regs[telescope_2])
            from_modulation = set(bsl_regs[baseline])
            if from_telescopes != from_modulation:
                log.error(
                    f"    Mismatch in region mapping for {baseline_name}: "
                    f"{from_telescopes} vs {from_modulation}"
                )
                raise ValueError("Baseline mapping consistency check failed.")
            bsl_to_reg[baseline] = order_abcd_regions(
                bsl_regs[baseline], detector_order
            )
            bsl_to_tel[baseline] = (telescope_1, telescope_2)

    product = P2VMPreprocProduct(
        spec_tel=spec_tel,
        spec_bsl=spec_bsl,
        spec_wavesc=spec_wavesc,
        spec_flat=model.aligned_flat,
        tel_regs=tel_regs,
        bsl_regs=bsl_regs,
        bsl_to_reg=bsl_to_reg,
        bsl_to_tel=bsl_to_tel,
        wl_grid=model.wavelength_grid,
    )
    write_product(config.output_dir / "p2vm_preproc.npz", product)
    log.info("Preprocessed P2VM and FLAT data saved.")
    log.info("--- Step: PREPROC P2VM [DONE] ---")
    log.info("")
    return product


def run_preproc_target(
    config: PipelineConfig, target_name: str
) -> ObjectPreprocProduct:
    """Extract and resample one science target cube."""
    if target_name not in config.targets:
        available = ", ".join(config.targets)
        raise ValueError(f"Unknown target {target_name!r}; available targets: {available}")
    log.info(f"--- Step: PREPROC {target_name} ---")
    model = _extraction_model(config)
    target = config.targets[target_name]
    dark = (
        model.flat.dark_map
        if target.dark is None
        else _mean_cube(config, target.dark, target.frames)
    )
    spec = _extract_cube(
        config,
        target.data,
        target.frames,
        dark,
        model,
    )
    product = ObjectPreprocProduct(
        spec=spec,
        spec_flat=model.aligned_flat,
        wl_grid=model.wavelength_grid,
    )
    write_product(config.output_dir / f"{target_name}_preproc.npz", product)
    log.info(f"Preprocessed target: {target_name}")
    log.info(f"--- Step: PREPROC {target_name} [DONE] ---")
    log.info("")
    return product


def _extraction_model(config: PipelineConfig) -> _ExtractionModel:
    flat = read_product(config.output_dir / "flat.npz", FlatProduct)
    wave = read_product(config.output_dir / "wave.npz", WaveProduct)
    parameters = config.parameters.resampling
    wl_grid = np.linspace(
        parameters.wavelength_min_um,
        parameters.wavelength_max_um,
        parameters.wavelength_channels,
    )
    spec_flat_detector = extract_spec_sparse(
        flat.flat_map[None, :, :], flat.profile_ys, flat.profile_xs
    )
    spec_flat_detector = np.clip(spec_flat_detector, a_min=1e-8, a_max=None)
    spec_flat = 1 / twopx_interp(1 / spec_flat_detector, wave.wave_map, wl_grid)
    return _ExtractionModel(
        flat=flat,
        wave_map=wave.wave_map,
        detector_flat=spec_flat_detector,
        aligned_flat=spec_flat,
        wavelength_grid=wl_grid,
    )


def _mean_cube(config: PipelineConfig, path: Path, frames: FrameRange) -> NDArray:
    log.info(f"    Building dark map from {path}...")
    selection = CubeSelection(frames.slice, config.crop.y.slice, config.crop.x.slice)
    return frame_mean(iter_cube(path, selection, config.chunk_frames))


def _extract_cube(
    config: PipelineConfig,
    path: Path,
    frames: FrameRange,
    dark: NDArray,
    model: _ExtractionModel,
) -> NDArray:
    log.info(f"    Extracting spectra from {path}...")
    selection = CubeSelection(frames.slice, config.crop.y.slice, config.crop.x.slice)
    aligned_chunks = []
    for data in iter_cube(path, selection, config.chunk_frames):
        spec = extract_spec_sparse(
            data - dark,
            model.flat.profile_ys,
            model.flat.profile_xs,
        )
        spec_aligned = twopx_interp(
            spec / model.detector_flat,
            model.wave_map,
            model.wavelength_grid,
        )
        spec_aligned *= model.aligned_flat
        aligned_chunks.append(spec_aligned)
    if not aligned_chunks:
        raise ValueError(f"Selected FITS cube contains no frames: {path}")
    return np.concatenate(aligned_chunks, axis=1)


@njit(parallel=True, fastmath=True, cache=True)
def extract_spec_cpu(cube: NDArray, profile_map: NDArray) -> NDArray:
    F, Y, X = cube.shape
    O = profile_map.shape[0]

    out = np.zeros((O, F, X), dtype=np.float32)

    # Reordered loops: O -> F -> Y -> X
    for o in prange(O):
        for f in range(F):
            for y in range(Y):
                for x in range(X):
                    pixel_val = cube[f, y, x]
                    mask_val = profile_map[o, y, x]
                    out[o, f, x] += pixel_val * mask_val
    return out


@njit(parallel=True, fastmath=True, cache=True)
def extract_spec_sparse(cube: NDArray, ys: NDArray, xs: NDArray) -> NDArray:
    """
    Extract spectra from a cube using sparse pixel indices per output channel.

    Args:
        cube (NDArray): Input cube with shape `(n_frame, ny, nx)`.
        ys (NDArray): Per-output Y indices; negative values are invalid sentinels.
        xs (NDArray): Per-output X indices; negative values are invalid sentinels.

    Returns:
        NDArray: Extracted spectra with shape `(n_reg, n_frame, nx)`.
    """
    n_frame, _, nx = cube.shape
    n_reg = len(ys)
    output = np.zeros((n_reg, n_frame, nx), dtype=np.float32)

    for o in prange(n_reg):
        y_idx = ys[o]
        x_idx = xs[o]
        for f in range(n_frame):
            for i in range(len(y_idx)):
                x = x_idx[i]
                y = y_idx[i]
                if x < 0 or y < 0:
                    continue
                output[o, f, x] += cube[f, y, x]
    return output


@njit(parallel=True, cache=True)
def twopx_interp(
    spec: NDArray, wave_map: NDArray, wl_grid: NDArray
) -> NDArray:
    """
    Resample spectra from detector pixels onto a uniform wavelength grid.

    Args:
        spec (NDArray): Input spectra with shape `(n_reg, n_frame, n_pixels)`.
        wave_map (NDArray): Wavelength at each detector pixel with shape `(n_reg, n_pixels)`.
        wl_grid (NDArray): Target wavelength grid with shape `(n_wave,)`.

    Returns:
        NDArray: Resampled spectra with shape `(n_reg, n_frame, n_wave)`.
    """
    n_reg, n_frame, _ = spec.shape
    n_wave = len(wl_grid)

    wn_map = 1.0 / wave_map
    wn_grid = 1.0 / wl_grid

    out = np.zeros((n_reg, n_frame, n_wave))

    for r in prange(n_reg):
        # We reverse (::-1) because np.interp expects increasing X values
        # and wavenumber usually decreases as pixel index increases.
        x_in = wn_map[r, ::-1]

        for f in range(n_frame):
            y_in = spec[r, f, ::-1]
            out[r, f, :] = np.interp(wn_grid, x_in, y_in)

    return out
