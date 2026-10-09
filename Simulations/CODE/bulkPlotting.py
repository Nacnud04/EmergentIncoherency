import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv
from scipy.integrate import simpson



# folder lists
Facet_folders = ["F020", "F050", "F100", "F200"]
Hurst_folders = ["H05"]
RMSH_folders  = ["H000000", "H001152"]
SIGLs         = [0.0, 0.025]

def load_datacube(ffolder, hfolder, rfolder):

    radar_paths = np.sort(
                    glob.glob(f"Radargrams/{ffolder}/{hfolder}/{rfolder}/rdr*/rdrgrm.npy")
                  )

    datacube = np.stack([np.load(p) for p in radar_paths], axis=-1) 

    # convert data cube into power
    datacube = np.abs(datacube) ** 2

    return datacube


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

# iterate over all combinations


for sig_l, rfolder in zip(SIGLs, RMSH_folders):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for ffolder in Facet_folders:
        for hfolder in Hurst_folders:

            fs    = int(ffolder[1:]) / 10
            hurst = int(hfolder[1:]) / 100
            rmsH  = int(rfolder[1:]) / 10000

            print(f"Loading in for {ffolder}/{hfolder}/{rfolder}")

            # first load in the datacube and get power at surface
            cube  = load_datacube(ffolder, hfolder, rfolder)
            PrNUM = extract_surface_return(cube)
            PrAVG = np.mean(PrNUM, axis=1)

            # get analytic power
            PrANA = Pr(Pt, Gt, hurst, sig_l, Hs/lam)

            # get error
            PrERR = 10*np.log10(PrAVG) - 10*np.log10(PrANA)

            ax.semilogx(Hs, PrERR, label=f"Facet size: {fs}", linewidth=1)

    plt.ylabel("Power Error [dB]")
    plt.xlabel("Altitude [m]")
    plt.title(f"Fresnel Zone w/ "+r"$\sigma_\lambda=$"+f"{sig_l:0.3f} & H={hurst:0.2f}")
    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.legend()
    plt.show()
