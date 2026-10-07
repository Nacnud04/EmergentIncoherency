import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

hurst_folder = "H40"
rms_folder = "H001256"
#rms_folder = "H003265"

radar_paths = np.sort(glob.glob(f"Radargrams/{hurst_folder}/{rms_folder}/rdr*/rdrgrm.npy"))
data = np.stack([np.load(p) for p in radar_paths], axis=-1)

Hs = np.logspace(np.log10(1000e3), np.log10(25e3), data.shape[1])

# linear power from complex voltage
P = np.abs(data[50, :, :])**2
P_mean = np.mean(P, axis=1)

breakpoint = 600e3#520e3

x_fits, y_fits = [], []
ms = []
r2s = []

for mask in (Hs < breakpoint, Hs >= breakpoint):

    x = Hs[mask]
    y = P_mean[mask]

    m, b = np.polyfit(np.log10(x), np.log10(y), 1)
    print(f"slope = {m:.4f}")

    y_fit = 10 ** (m * np.log10(x) + b)

    ss_res = np.sum((y - y_fit) ** 2)    
    ss_tot = np.sum((y - np.mean(y)) ** 2)    
    r2 = 1 - ss_res / ss_tot
    print(f"R^2 = {r2:.4f}")

    x_fits.append(Hs[mask])
    y_fits.append(y_fit)
    ms.append(m)
    r2s.append(r2)

# convert to dB for plotting
P_db = 10 * np.log10(P)
P_mean_db = 10 * np.log10(P_mean)

# attempt to compute analytic solution
# scaling factor

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

Hurst = 0.4
rms_H_m = 5.0# 13.0
rms_P_m = 10e3
lam   = 4.99654

# convert to for baseline profile of lambda
sig_lam = (rms_H_m / lam) / ((rms_P_m / lam)**Hurst)
print(sig_lam)

Ls = L(Hurst, sig_lam, Hs/lam, N)
Fh = (1 / ((Hs / 5.0)**2)) * Ls
Pr = (100 * (10**0.73)**2 * Fh) / (4 * np.pi)**2

plt.figure()
for i in range(P_db.shape[-1]):
    plt.semilogx(Hs, P_db[:, i], color="black", alpha=0.1)
plt.semilogx(Hs, P_mean_db, color="red", label="mean")
colors = ("blue", "cyan")
for xfit, yfit, m, r, clr in zip(x_fits, y_fits, ms, r2s, colors):
    plt.semilogx(xfit, 10 * np.log10(yfit), color=clr, linestyle="--", label=f"fit slope={m:.3f} r={r:.2f}")

plt.semilogx(Hs, 10 * np.log10(Pr), color="magenta", label="Analytic")

plt.xlabel("Altitude [m]")
plt.ylabel("Power [dB]")
plt.xlim(np.min(Hs), np.max(Hs))
plt.legend()
plt.tight_layout()
plt.show()

plt.semilogx(Hs, P_mean_db - 10*np.log10(Pr))
plt.show()
