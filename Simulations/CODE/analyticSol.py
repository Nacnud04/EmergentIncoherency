import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import simpson

N = (1 - jv(0, 2*np.pi)) / (2*np.pi)

def L(H, sig_l, h_l, N, dx=1e-5):
    x = np.linspace(0.0, 1.0, int(np.ceil(1/dx)) + 1)
    sig_l = np.asarray(sig_l)[:, None]
    x2 = x[None, :]**2

    integrand = jv(1, 2*np.pi*x[None, :]) * np.exp(
        - (4*np.pi)**(2*(1+H)) * sig_l**2 * (0.5 * h_l * x2)**H
    )

    integral = simpson(integrand, x=x, axis=1)
    return integral / N

h_l = 8000
sig_l = np.linspace(0, 0.35, 250)

Hs = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

Fs = []

for Hurst in Hs:
    print(f"Evaluating at Hurst exponent: {Hurst}")
    Ls = L(Hurst, sig_l, h_l, N)
    Fh = (1 / (h_l**2)) * Ls
    Fs.append(Fh)

for F in Fs:
    plt.plot(sig_l, 10 * np.log10(F), color="black", linewidth=1)

plt.ylim(-150, -60)
plt.xlim(0, 0.35)

plt.grid(linestyle=":", color="grey")

plt.xlabel(r"$\sigma_\lambda$")
plt.ylabel(r"Power [dB]")

plt.title(r"Altitude $8000\lambda$")

plt.show()