"""Group-delay estimator shared by P2VM calibration and science reduction."""

import numpy as np
from numpy.typing import NDArray


def estimate_group_delay(
    visdata: NDArray,
    wavelength: NDArray,
    search_range: tuple[float, float] = (-30e-6, 30e-6),
    n_newton: int = 5,
) -> NDArray:
    """Locate the coherent-sum maximum and refine its optical delay with Newton steps.

    Parameters
    ----------
    visdata : complex ndarray, shape (sample, wavelength)
        Coherent spectra or dimensionless phasors. Each row is fitted independently;
        all wavelength channels enter equally, without masking or noise weights.
    wavelength : ndarray, shape (wavelength,)
        Wavelengths in metres, matching the final axis of visdata.
    search_range : tuple of float, optional
        Coarse delay interval [lower, upper) in metres; default +/-30 micrometres.
        The fixed grid spacing is 1e-8 m. This is an initialization interval,
        not a bound on the returned delay.
    n_newton : int, optional
        Number of unconstrained Newton updates of squared coherent-sum amplitude;
        default 5. A zero Hessian leaves that sample's current delay unchanged.

    Returns
    -------
    group_delay_m : ndarray, shape (sample,)
        Delay maximizing abs(sum(C(lambda) * exp(-2j*pi*GD/lambda))) locally,
        in metres, with this explicit phase/sign convention.

    Notes
    -----
    This estimator does not certify a detected fringe. An all-zero spectrum
    returns the lower search edge, and nonfinite inputs can yield nonfinite
    results. Interpret these values together with signal validity diagnostics.
    """
    k = -2j * np.pi / wavelength
    w = -2 * np.pi / wavelength

    gd_grid = np.arange(search_range[0], search_range[1], 1e-8)
    phasors_grid = np.exp(gd_grid[:, None] * k[None, :])
    coherence = np.abs(np.tensordot(visdata, phasors_grid, axes=([-1], [-1])))

    best = np.argmax(coherence, axis=-1)
    gd_current = gd_grid[best]

    for _ in range(n_newton):
        gd_exp = gd_current[:, None]
        rotated = visdata * np.exp(1j * w * gd_exp)
        sum_0 = np.sum(rotated, axis=-1)
        sum_1 = np.sum(rotated * (1j * w), axis=-1)
        sum_2 = np.sum(rotated * (-(w**2)), axis=-1)

        sum_0_conjugate = np.conj(sum_0)
        gradient = 2 * np.real(sum_1 * sum_0_conjugate)
        hessian = 2 * np.real(sum_2 * sum_0_conjugate + sum_1 * np.conj(sum_1))
        difference = np.zeros_like(gradient)
        valid = hessian != 0
        difference[valid] = gradient[valid] / hessian[valid]
        gd_current = gd_current - difference

    return gd_current
