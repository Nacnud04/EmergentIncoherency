import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv
from scipy.integrate import simpson

from chris_fractal import fractal_loss

# folder lists
#Facet_folders = ["F020", "F050", "F100", "F200"]
Facet_folders = ["F200"]
Hurst_folders = ["H05"]
#RMSH_folders  = ["H000000", "H001152", "H002305", "H003457", "H004610"]
#SIGLs         = [0.0, 0.025, 0.05, 0.075, 0.1]
RMSH_folders = ["H001152"]
SIGLs        = [0.025]

def extract_surface_return(datacube):
    smpl = datacube.shape[0] // 2
    return datacube[smpl, :, :]

def L(H, sig_l, h_l, N, dx=1e-5):
    x = np.linspace(0.0, 1.0, int(np.ceil(1/dx)) + 1)
    h_l = np.asarray(h_l)[:, None]
    x2 = x[None, :]**2

    integrand = jv(1, 2*np.pi*x[None, :]) * np.exp(
        - (4*np.pi)**(2*(1+H)) * sig_l**2 * (0.5 * h_l * x2)**H
    )

    integral = simpson(integrand, x=x, axis=1)
    return integral / N

def Pr(Pt, G, H, sig_l, h_l):
    N  = (1 - jv(0, 2*np.pi)) / (2*np.pi)
    Ls = L(H, sig_l, h_l, N)
    Fh = (1 / (h_l**2)) * Ls
    Pr = (Pt * (10**(G/10))**2 * Fh) / (4 * np.pi)**2
    return Pr

Pt = 100
Gt = 7.3
Hs = np.logspace(np.log10(1000e3), np.log10(25e3), 1000)
lam = 4.996
c = 299792458

# RMS height at 1 meter:
RMS_1m = np.array([int(f[1:])/1e4 for f in RMSH_folders])

def Pr_Chris(Pt, G, H, sig_1m, Heights, lam):
    N  = (1 - jv(0, 2*np.pi)) / (2*np.pi)
    Ls = fractal_loss(sig_1m, H, c / lam, Heights, lag=1)
    Fh = (1 / ((Heights/lam)**2)) * Ls
    Pr = (Pt * (10**(G/10))**2 * Fh) / (4 * np.pi)**2
    return Pr

def extractAndGetDifference(ffolder, hfolder, rfolder, pdir):

    fs    = int(ffolder[1:]) / 10
    hurst = int(hfolder[1:]) / 100
    rmsH  = int(rfolder[1:]) / 10000

    print(f"Loading in for {ffolder}/{hfolder}/{rfolder}")

    npy_file = os.path.join(pdir, f"{ffolder}{hfolder}{rfolder}.npy")
    cube = np.load(npy_file)

    PrNUM = extract_surface_return(cube)
    PrAVG = np.mean(PrNUM, axis=1)

    PrANA = Pr(Pt, Gt, hurst, sig_l, Hs/lam)
    PrANA_CH = Pr_Chris(Pt, Gt, hurst, r1m, Hs, lam)
    PrERR = 10*np.log10(PrAVG) - 10*np.log10(PrANA_CH)

    return PrERR

# same subdirectory for inputs and outputs

backscatter_models = ("Lossless", "CoherentLoss", "CoherentLossSqrt", "SimpleBackscatter")

for hfolder in Hurst_folders:
    for sig_l, rfolder, r1m in zip(SIGLs, RMSH_folders, RMS_1m):
        if sig_l == 0:
            continue
        for ffolder in Facet_folders:

            fig, ax = plt.subplots(1, 1, figsize=(14, 7))

            fs = int(ffolder[1:]) / 10
            hurst = int(hfolder[1:]) / 100

            # iterate over backscatter models
            for bmdl in backscatter_models:

                PLOT_DIR = f"Plots/{bmdl}"

                PrERR = extractAndGetDifference(ffolder, hfolder, rfolder, PLOT_DIR)
                
                ax.semilogx(Hs, PrERR, label=f"{bmdl}", linewidth=1)

            ax.set_ylabel("Power Error [dB]")
            ax.set_xlabel("Altitude [m]")
            ax.set_title(f"Fresnel Zone w/ " + r"$\sigma_\lambda=$" + f"{sig_l:0.3f} & H={hurst:0.2f} & facet size={fs}")
            ax.axhline(0, linestyle="--", color="black", linewidth=1)
            ax.legend()

            out_png = os.path.join("Plots/Comparison", f"{hfolder}{rfolder}{ffolder}.png")
            plt.savefig(out_png, dpi=300)
            plt.close()

