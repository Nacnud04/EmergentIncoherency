import sys

sys.path.append("../..//EuropaRaypaths/src/PYTHON")
import param_gen       as pg


recpar = {
    "rx_window_m": 0.5e3,         # receive window size [m]
    "rx_sample_rate": 60e6,        # receive sample rate [Hz]
    "rx_window_position_file": "Simpar/rx_window_positions.txt",
}

sourcepar = {
    "ns": 1000,                    # source count            [.]
    "source_path_file": "Simpar/source_path.txt",
    "aperture": 70,
}

sig_l = 0.025
lam = 4.996
surf_width  = 8200        # simulation width [m]
facet_sizes = [20, 10, 5, 2]
RMSh_vals   = [0, 0.1152, 0.2305, 0.3457, 0.4610]
sigl_vals   = [0, 0.025, 0.05, 0.075, 0.1]
Hurst_exps  = [0.05, 0.1, 0.2]

for fs in facet_sizes:
    for Hurst in Hurst_exps:
        for rmsh, sig_l in zip(RMSh_vals, sigl_vals):
            
            fdir = f"F{int(fs*10):03d}"
            hdir = f"H{int(Hurst*100):02d}"
            rdir = f"H{int(rmsh*10000):06d}"

            surfdir = f"Surfaces/{fdir}/{hdir}/{rdir}"

            domainpar = {
                "ox": -4100,
                "oy": -4100,
                "oz": 0,
                "fs": fs,
                "nx": int(surf_width) / fs,
                "ny": int(surf_width) / fs,
            }

            fct = domainpar['fs']
            rms_fct = sig_l * lam * (fct / lam)**Hurst

            otherpar = {
                'lossless': False,
                "fresnel": True,
                "rms_height": rms_fct,
                "roughness_model": "specular",
            }

            roughness_models = ("specular", "coherent_specular", "simple_backscatter")

            for rModel in roughness_models:
                otherpar["roughness_model"] = rModel 

                # first export basic single example stuff
                params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)
                pg.export_params(params, f"VHF_{rModel}", directory=surfdir)
                print(f"Exported VHF_{rModel} into directory {surfdir}")

# --- MAKE SOURCE PATH ---

maxZ = 25e3 # maximum altitude of source path [m]
minZ = 1000e3 # minimum altitude of source path [m]

sz = pg.vert_source_path(params, minZ, maxZ, "source_path", direct="Simpar/.", log=True)

# --- RX OPENING WINDOW FILE ---

pg.track_Z_rxwin(sz, 250, "rx_window_positions", direct="Simpar/.")
