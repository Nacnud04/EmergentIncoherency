import numpy as np
import matplotlib.pyplot as plt

# QUESTION: 
# what is the angle to the edge of the fresnel zone?

# --- PARAMETERS ---

c = 299792458

Fs   = np.array([60e6, 20e6, 9e6, 2e6])
lams = c / Fs

Hs   = np.logspace(np.log10(1e3), np.log10(1000e3), 500)


# --- COMPUTE ANGLE TO EDGE OF FIRST FRESNEL ZONE ---

plt.rcParams["text.usetex"] = True

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for f, lam in zip(Fs, lams):

    # compute angle to edge of fresnel zone
    theta = np.arctan(np.sqrt(lam / (2 * Hs)))

    ax[0].semilogx(Hs / 1e3, np.degrees(theta), label=f"{f/1e6:.0f} MHz")

    # compute the error introduced by q
    error      = np.abs(np.cos(theta) - theta) / np.cos(theta)
    ax[1].semilogx(Hs / 1e3, error, label=f"{f/1e6:.0f} MHz")

ax[0].set_ylabel(r"$\theta_f$ [$^\circ$]")

# add overlay for typical spacecraft altitudes
for a in ax:
    a.axvspan(25, 1000, color="cyan", alpha=0.1, label="REASON Altitudes")
    # https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2006JE002701
    a.axvspan(250, 315, color="red", alpha=0.1, label="SHARAD Altitudes")
    # https://www2.mps.mpg.de/de/projekte/mars-express/marsis/index_print.html
    a.axvspan(250, 800, color="magenta", alpha=0.1, label="MARSIS Altitudes")

    a.set_xlim(1, 1000)

    a.legend()
    a.grid()
    a.set_xlabel("Altitude [km]")

ax[1].set_ylabel("Error")
ax[1].set_title(r"Error in $\cos(\theta_f)$ Approximation")
ax[0].set_title(r"Angle to Edge of First Fresnel Zone")

plt.savefig("FresnelAngle.pdf")

plt.show()