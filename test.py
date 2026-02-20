from astropy.io import fits

from oifits_schema import GRAVITY_FT, OI

with fits.open(
    "~/data/kb230025_ep1/reduced/GRAVI.2023-06-01T03:26:38.962_dualscivis.fits"
) as hdul:
    oi = OI.load(hdul, GRAVITY_FT)
    print(oi)

    import matplotlib.pyplot as plt
    plt.style.use("./zexwu.mplstyle")
    fig, ax = plt.subplots(figsize=(6, 6))
    for tri in range(4):
        ax.plot(oi.wavelength.eff_wave*1e6, oi.t3.t3phi[tri::4, :].mean(axis=0), "o-")
    plt.show()
