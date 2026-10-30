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

plt.plot(RMSs / lam, gamma_coh_dB, label=r"$\exp\left(-4\left(ks\cos\theta\right)^2\right)$")
plt.plot(RMSs / lam, gamma_coh_2_dB, label=r"$\exp\left(-2\left(ks\cos\theta\right)^2\right)$")
plt.plot(RMSs / lam, rho_dB, label=r"$\exp\left(-g^2\right)\mathcal{I}_0\left(\frac{g^2}{2}\right)^2$")
plt.axvline(0.125, color="red", linestyle="--", label="Rayleigh Criterion for Roughness")
plt.ylim(-50, 0)
plt.xlim(0, 1)
plt.xlabel(r"RMS Height [$\lambda$]")
plt.ylabel(r"Loss [dB]")
plt.title(r"Various Backscatter Models $\theta_i=0$") 
plt.legend()
plt.savefig("Plots/backscatterModels.pdf")
plt.show()

