import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import simpson

N = (1 - jv(0, 2*np.pi)) / (2*np.pi)

def L(H, sig_l, h_l, N, dx=1e-5):
    x = np.linspace(0.0, 1.0, int(np.ceil(1/dx)) + 1)
    h_l = np.asarray(h_l)[:, None]
    x2 = x[None, :]**2

    integrand = jv(1, 2*np.pi*x[None, :]) * np.exp(
        - (4*np.pi)**(2*(1+H)) * sig_l**2 * (0.5 * h_l * x2)**H
    )

    integral = simpson(integrand, x=x, axis=1)
    return integral / N

lam = 4.99654

# altitudes
alts = np.logspace(np.log10(25e3), np.log10(1000e3), 250)
h_l  = alts / lam

# Hurst
Hurst = 0.05

# rms height (in lam at length lam)
sig_ls = np.linspace(0, 0.1, 5)
print(sig_ls)

# get the rms height at an equivalent 10 km profile
bsls = (1, 10e3)
for bsl in bsls:
    sig_10 = sig_ls * lam * (bsl / lam) ** Hurst
    print(f"RMS {bsl} m: {sig_10}")

# what would sig_l be for the 5 meter rms height?
sig_l_extreme = (5.0 / (bsl / lam) ** Hurst) / lam
print(f"Sig_l @ a 5 m RMS over 10 km: {sig_l_extreme}") 

# solve
Fs = []
for sig_l in sig_ls:
    Ls = L(Hurst, sig_l, h_l, N)
    Fh = (1 / (h_l**2)) * Ls
    Fs.append(Fh)
Fs = np.array(Fs)

# get const power scaling
Pr = (100 * (10**0.73)**2 * Fs) / (4 * np.pi)**2

# plot
for P, sig_l in zip(Pr, sig_ls):
    plt.semilogx(alts, 10 * np.log10(P), label=f"{sig_l:0.3f}")
    # get the power fall off for each
    m, b = np.polyfit(np.log10(alts), np.log10(P), 1)
    print(f"Decay w/ {sig_l:0.3f} is {b:0.1f}")
    # solve and plot the fit
    #y_fit = 10 ** (m * np.log10(alts) + b)
    #plt.semilogx(alts, 10 * np.log10(y_fit), label=f"fit: {sig_l:0.3f}", linestyle=":")
plt.legend()
plt.grid()
plt.xlabel("Altitude [m]")
plt.ylabel("Power [dB]")
plt.show()