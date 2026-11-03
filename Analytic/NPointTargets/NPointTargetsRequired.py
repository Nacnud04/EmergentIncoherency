import numpy as np
import matplotlib.pyplot as plt

# set up constant parameters
lam = 4.996      # wavelength [m]
rad_cor = 0.3      # radians for phase correlation

# set up range parameters
sig_ls = np.linspace(0.025, 0.1, 250)    # RMS height [lambda]
Hursts = np.linspace(0, 1, 250)          # Hurst exponent

# evaluate the minimum target spacing for adequate phase sampling
dl = lam * (rad_cor / (4 * np.pi * sig_ls[:, None])) ** (1 / Hursts[None, :])

# how many facets do you need to represent the fresnel zone?
R = 25e3
Ntargets = (np.pi * lam * R) / (2 * dl**2)
Ntargets_dB = 10 * np.log10(Ntargets)


fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ext = [0, 1, 0.025, 0.1]
im0 = ax[0].imshow(dl, extent=ext, origin="lower", aspect="auto", cmap="viridis", vmax=10, vmin=0.001)
cs = ax[0].contour(dl, levels=[0.001, 0.1, 1, 2.5, 5, 10, 100], colors="red", linewidths=1, extent=ext)
ax[0].clabel(cs, inline=True, fontsize=8, fmt="%.3f")
plt.colorbar(im0, ax=ax[0], label="Minimum Facet Spacing [m]")

im1 = ax[1].imshow(Ntargets_dB, extent=ext, origin="lower", aspect="auto", vmax=60, vmin=30)
cs = ax[1].contour(Ntargets_dB, levels=[30, 40, 50, 60], colors="red", linewidths=1, extent=ext)
ax[1].clabel(cs, inline=True, fontsize=8, fmt="%.0f dB")
plt.colorbar(im1, ax=ax[1], label="Number of Facets in Fresnel Zone [dB]")

for a in ax:
    a.set_xlabel("Hurst Exponent")
    a.set_ylabel("RMS Height [$\lambda$]")

plt.tight_layout()

plt.show()