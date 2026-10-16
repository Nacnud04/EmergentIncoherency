import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

# --- PARAMETER SPACE ---
# general parameters
lam = 4.996
k   = (2 * np.pi) / lam

# incidence angles
#thetas = np.radians(np.linspace(0, 90, 200))
theta = 0
# within-facet RMS height
RMSs   = np.linspace(0, 5, 200)

# make parameter meshgrid
#thss, Sss = np.meshgrid(thetas, RMSs)


# --- SOLVE APPROACH 1: COHERENT LOSS IN THE SPECULAR DIRECTION ---

# exponent
phi = k * RMSs * np.cos(theta)
# solve
gamma_coh = np.exp(-4 * phi**2)
# move into dB
gamma_coh_dB = 10 * np.log10(gamma_coh)


# --- SOLVE APPROACH 2: ACCOUNT FOR COHERENT AND INCOHERENT USING SIMPLE MODEL ---
# Boithias 1987
# https://doi.org/10.1029%2F2004JB003222

g = (4 * np.pi * RMSs) / lam
rho = np.exp(-1 * g**2) * np.i0(g**2 / 2)**2
rho_dB = 10 * np.log10(rho)


# --- SOLVE APPROACH 3: COHERENET LOSS IN THE SPECULAR DIRECTION, SCALED DOWN ---

# exponent
phi = k * RMSs * np.cos(theta)
# solve
gamma_coh_2 = np.exp(-2 * phi**2)
# move into dB
gamma_coh_2_dB = 10 * np.log10(gamma_coh_2)

# --- PLOT ---

plt.plot(RMSs / lam, gamma_coh_dB, label=r"$\exp\left(-4ks\cos\theta\right)$")
#plt.plot(RMSs / lam, gamma_coh_2_dB, label=r"$\exp\left(-2ks\cos\theta\right)$")
plt.plot(RMSs / lam, rho_dB, label=r"$\exp\left(-g^2\right)\mathcal{I}_0\left(\frac{g^2}{2}\right)^2$")
plt.axvline(0.125, color="red", linestyle="--", label="Rayleigh Criterion for Roughness")
plt.ylim(-50, 0)
plt.xlim(0, 1)
plt.xlabel(r"RMS Height [$\lambda$]")
plt.ylabel(r"Loss [dB]")
plt.legend()
plt.show()

"""
extent = [np.degrees(thetas[0]), np.degrees(thetas[-1]), RMSs[-1]/lam, RMSs[0]/lam]

im = plt.imshow(gamma_coh_dB, extent=extent, aspect='auto', vmin=-200)

# Add contour lines at regular dB intervals
levels = np.concatenate([np.arange(-200, 0, 20), [-10, -5, -1]])  # Contours every 20 dB from -200 to 0
contours = plt.contour(gamma_coh_dB, levels=levels, extent=extent, 
                       colors='red', linewidths=0.5, alpha=0.6, origin='upper')
plt.clabel(contours, inline=True, fontsize=8, fmt='%d dB', colors="red")

plt.ylabel(r"RMS Height [$\lambda$]")
plt.xlabel(r"Incident Angle [$\theta$]")
plt.colorbar(im)  # Explicitly pass the imshow image to colorbar
plt.title(r"$\exp\left(-4ks\cos\theta\right)$")
plt.savefig("Plots/CoherentLoss.pdf")
plt.show()


im = plt.imshow(rho_dB, extent=extent, aspect='auto', vmin=-200)

# Add contour lines at regular dB intervals
levels = np.concatenate([np.arange(-200, 0, 20), [-10, -5, -1]])  # Contours every 20 dB from -200 to 0
contours = plt.contour(rho_dB, levels=levels, extent=extent, 
                       colors='red', linewidths=0.5, alpha=0.6, origin='upper')
plt.clabel(contours, inline=True, fontsize=8, fmt='%d dB', colors="red")

plt.ylabel(r"RMS Height [$\lambda$]")
plt.xlabel(r"Incident Angle [$\theta$]")
plt.colorbar(im)  # Explicitly pass the imshow image to colorbar
#plt.title(r"$\exp\left(-4ks\cos\theta\right)$")
plt.savefig("Plots/SimplifiedLoss.pdf")
plt.show()
"""