import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from tqdm import tqdm

import roughSynth as rs


# --- SURFACE PARAMETERS --- 

N = 400
M = 400

L_phys = 1000.0

dx = L_phys / N
dy = L_phys / M

sigmas = np.linspace(0, 4, 100)
corr_length = 50.0

lam = 15

# compute gain
G_t = 10**(0.73)
# this just uses the maximum gain of the facet
G_f = (4 * np.pi * dx * dy) / (lam ** 2)

# transmit power 
Pt = 20

# altitude
h = 25e3

# fresnel zone radius
fr = np.sqrt(lam * h / 2)
print(f"Fresnel zone radius: {fr:.2f} m")

def q(th):
    return (4 * np.pi / lam) * np.cos(th)

def friis(P, rho):
    R = np.sqrt(h**2 + rho**2)
    return (P * G_t * G_f * lam**2) / ((4 * np.pi)**2 * R**2)

def w(P, rho):
    R = np.sqrt(h**2 + rho**2)

    P_f = friis(P, rho)
    P_r = friis(P_f, rho)

    # Amplitude attenuation
    w_amp = np.sqrt(P_r)

    # Two-way propagation phase relative to nadir
    phase = -4 * np.pi * (R - h) / lam

    # Complex point-target weight
    w_val = w_amp * np.exp(1j * phase)

    # First Fresnel zone
    w_val[rho > fr] = 0.0

    return w_val


Pcs = []
Pns = []
Prs = []
Fracs = []

for sigma in tqdm(sigmas):

    # --- GENERATE SURFACE ---

    field, psd_theory, cov_theory = rs.gaussian_surface(
        N, M,
        dx, dy,
        sigma=sigma,
        corr_length=corr_length
    )

    # flatten covariance to 1D
    rho_theory_pix, cov1d_theory, _ = rs.radial_covariance(cov_theory)
    rho_theory = rho_theory_pix * dx

    # get rms height from covariance
    sig_h_sq = cov1d_theory[0]

    # solve for phase of point target phasors
    thetas = np.arctan(rho_theory / h)
    K = np.exp(-1 * q(thetas)**2 * (sig_h_sq - cov1d_theory)) - np.exp(-1 * q(thetas)**2 * sig_h_sq)


    # --- COMPUTE THEORETICAL PHASOR COVARIANCE ---

    # Use constant q (nadir incidence)
    q0 = q(0.0)

    # Mean phasor
    mu = np.exp(-0.5 * q0**2 * sig_h_sq)

    # Construct full lag grid for the covariance kernel
    lag_x = np.arange(-(N-1), N) * dx
    lag_y = np.arange(-(M-1), M) * dy

    XX_lag, YY_lag = np.meshgrid(lag_x, lag_y)
    rho_lag = np.sqrt(XX_lag**2 + YY_lag**2)

    # Interpolate theoretical surface covariance onto the lag grid
    C_lag = np.interp(
        rho_lag.ravel(),
        rho_theory,
        cov1d_theory,
        left=cov1d_theory[0],
        right=0.0
    ).reshape(rho_lag.shape)

    # Phasor covariance kernel
    K_lag = (
        np.exp(-q0**2 * (sig_h_sq - C_lag))
        - np.exp(-q0**2 * sig_h_sq)
    )

    # --- COMPUTE SPATIAL WEIGHTS ---

    # Physical coordinates of facets
    x = (np.arange(N) - (N-1)/2) * dx
    y = (np.arange(M) - (M-1)/2) * dy

    XX, YY = np.meshgrid(x, y)
    rho_grid = np.sqrt(XX**2 + YY**2)

    # Two-way amplitude weights
    W = w(Pt, rho_grid)

    # --- COHERENT POWER ---

    P_c = np.abs(np.sum(W * mu))**2

    # --- INCOHERENT POWER ---

    # Convolve weights with phasor covariance
    KW = fftconvolve(np.conj(W), K_lag, mode="same")

    P_n = np.sum(W * KW)

    # --- TOTAL POWER ---

    P_total = P_c + P_n

    Pcs.append(P_c)
    Pns.append(P_n)
    Prs.append(P_total)
    Fracs.append(P_n / P_total if P_total != 0 else 0)

    #print(f"Coherent power:   {P_c:.6e}")
    #print(f"Incoherent power: {P_n:.6e}")
    #print(f"Total power:      {P_total:.6e}")
    #print(f"Incoherent fraction: {P_n/P_total:.4%}")


# --- COMPARE ANALYTIC SOL WITH HAYNES ---

P_fresnel = (Pt * G_t * G_t * lam**2) / (2**4 * np.pi**2 * h**2)
P_fresnel_dB = 10 * np.log10(P_fresnel)

print(f"Haynes Fresnel Power: {P_fresnel_dB:.2sf} dB")

print(f"Analytic Fresnel Power: {10 * np.log10(np.abs(Prs[0])):.2f} dB")


# --- PLOT POWERS ---

plt.rcParams["text.usetex"] = True

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.rcParams['lines.linewidth'] = 1.0

Pcs_dB = 10 * np.log10(Pcs)
Pns_dB = 10 * np.log10(Pns)
Prs_dB = 10 * np.log10(Prs)

ax[0].plot(sigmas, Pcs_dB, label=r"$P_c$")
ax[0].plot(sigmas, Pns_dB, label=r"$P_n$")
ax[0].plot(sigmas, Prs_dB, label=r"$P_t$")

ax[0].set_ylabel("Power [dB]")
ax[0].set_ylim(np.min(Prs_dB), np.max(Prs_dB))

ax[1].plot(sigmas, Fracs, label=r"$P_n / P_t$", color="blue")
ax[1].plot(sigmas, 1 - np.array(Fracs), label=r"$P_c / P_t$", color="red")

ax[1].set_ylabel("Power Fraction")

for a in ax:
    a.set_xlabel(r"RMS Height [m]")
    a.grid()
    a.legend()

plt.show()
