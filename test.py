from astropy.io import fits

from oifits_schema import (GRAVITY_FT, OI_ARRAY, OI_FLUX, OI_T3, OI_VIS,
                           OI_WAVELENGTH)

with fits.open(
    "~/data/kb230025_ep1/reduced/GRAVI.2023-06-01T03:26:38.962_dualscivis.fits"
) as hdul:
    oi_arr  = OI_ARRAY      (hdul)
    oi_wave = OI_WAVELENGTH (hdul, GRAVITY_FT)
    oi_flux = OI_FLUX       (hdul, GRAVITY_FT)
    oi_vis  = OI_VIS        (hdul, GRAVITY_FT)
    oi_t3   = OI_T3         (hdul, GRAVITY_FT)

    print(oi_arr)
    print(oi_t3)
    print(oi_wave)
    print(oi_vis)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(oi_wave.eff_wave, oi_t3.t3phi[::4, :].mean(axis=0))
    plt.show()
