import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from scipy.special import jv
from scipy.integrate import simpson

from chris_fractal import fractal_loss


Pt = 100
Gt = 7.3
Hs = np.logspace(np.log10(1000e3), np.log10(25e3), 1000)
lam = 4.996
c = 299792458

backscatter_models = ("Lossless", "CoherentLoss", "CoherentLossSqrt", "SimpleBackscatter")
backscatter_styles = ("solid", "dotted", "dashed", "dashdot")

Facet_folders = ["F020", "F050", "F100", "F200"]
facet_colors  = ["blue", "red", "lime", "magenta"]

RMSH_folders  = ["H001152", "H002305", "H003457", "H004610"]
SIGLs         = [0.025, 0.05, 0.075, 0.1]

hfolder = "H05"
RMS_1m = np.array([int(f[1:]) / 1e4 for f in RMSH_folders])


def L(H, sig_l, h_l, N, dx=1e-5):
    x = np.linspace(0.0, 1.0, int(np.ceil(1 / dx)) + 1)
    h_l = np.asarray(h_l)[:, None]
    x2 = x[None, :] ** 2

    integrand = jv(1, 2*np.pi*x[None, :]) * np.exp(
        - (4*np.pi)**(2*(1+H)) * sig_l**2 * (0.5 * h_l * x2)**H
    )

    integral = simpson(integrand, x=x, axis=1)
    return integral / N


def Pr(Pt, G, H, sig_l, h_l):
    N  = (1 - jv(0, 2*np.pi)) / (2*np.pi)
    Ls = L(H, sig_l, h_l, N)
    Fh = (1 / (h_l**2)) * Ls
    return (Pt * (10**(G/10))**2 * Fh) / (4 * np.pi)**2


def Pr_Chris(Pt, G, H, sig_1m, Heights, lam):
    N  = (1 - jv(0, 2*np.pi)) / (2*np.pi)
    Ls = fractal_loss(sig_1m, H, c / lam, Heights, lag=1)
    Fh = (1 / ((Heights / lam)**2)) * Ls
    return (Pt * (10**(G/10))**2 * Fh) / (4 * np.pi)**2


# caches for analytic curves
prana_cache = {}
prana_ch_cache = {}


def extractAndGetDifference(ffolder, hfolder, rfolder, pdir, sig_l, r1m, hurst):
    npy_file = os.path.join(pdir, f"{ffolder}{hfolder}{rfolder}.npy")
    cube = np.load(npy_file, mmap_mode="r")

    # middle slice only
    PrAVG = cube[cube.shape[0] // 2].mean(axis=1)

    # cache by inputs that affect the result
    prana_key = (Pt, Gt, hurst, sig_l, lam, tuple(Hs))
    if prana_key not in prana_cache:
        prana_cache[prana_key] = Pr(Pt, Gt, hurst, sig_l, Hs / lam)

    prana_ch_key = (Pt, Gt, hurst, r1m, lam, tuple(Hs))
    if prana_ch_key not in prana_ch_cache:
        prana_ch_cache[prana_ch_key] = Pr_Chris(Pt, Gt, hurst, r1m, Hs, lam)

    PrANA_CH = prana_ch_cache[prana_ch_key]

    # only if you actually need PrANA, otherwise remove it
    # PrANA = prana_cache[prana_key]

    PrERR = 10 * np.log10(PrAVG) - 10 * np.log10(PrANA_CH)
    return PrERR


fig, ax = plt.subplots(len(SIGLs), figsize=(10, 18), sharex=True)

total = len(SIGLs) * len(Facet_folders) * len(backscatter_models)
pbar = tqdm(total=total, desc="Generating plots")

for i, (sig_l, rfolder, r1m) in enumerate(zip(SIGLs, RMSH_folders, RMS_1m)):
    hurst = int(hfolder[1:]) / 100

    for j, ffolder in enumerate(Facet_folders):
        for k, bmdl in enumerate(backscatter_models):
            PLOT_DIR = f"Plots/{bmdl}"

            PrERR = extractAndGetDifference(
                ffolder=ffolder,
                hfolder=hfolder,
                rfolder=rfolder,
                pdir=PLOT_DIR,
                sig_l=sig_l,
                r1m=r1m,
                hurst=hurst
            )

            ax[i].semilogx(
                Hs, PrERR,
                linewidth=1,
                linestyle=backscatter_styles[k],
                color=facet_colors[j]
            )

            pbar.update(1)

    ax[i].set_ylabel("Power Error [dB]")
    ax[i].set_title(
        f"Fresnel Zone w/ " + r"$\sigma_\lambda=$" + f"{sig_l:0.3f} & H={hurst:0.2f}",
        fontsize=10
    )
    ax[i].axhline(0, linestyle="--", color="black", linewidth=1)

pbar.close()

facet_sizes = np.array([2.0, 5.0, 10.0, 20.0])
flambda = facet_sizes / 5

color_handles = [
    Line2D([0], [0], color=facet_colors[j], lw=2, label=f"{flambda[j]:.1f}$\lambda$")
    for j in range(len(Facet_folders))
]

style_handles = [
    Line2D([0], [0], color="black", linestyle=backscatter_styles[k], lw=2, label=backscatter_models[k])
    for k in range(len(backscatter_models))
]

leg1 = ax[0].legend(handles=color_handles, title="Facet size", loc="upper right")
ax[0].add_artist(leg1)
ax[0].legend(handles=style_handles, title="Backscatter model", loc="upper center")

ax[-1].set_xlabel("Altitude [m]")

plt.tight_layout()
plt.savefig("Plots/bigPlotComparison.pdf")
plt.show()
