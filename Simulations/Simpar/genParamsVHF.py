import sys

sys.path.append("/mnt/c/Users/duby2704/OneDrive - UCB-O365/Documents/EuropaRaypaths/src/PYTHON")
import param_gen       as pg

domainpar = {
    "ox": -4100,
    "oy": -4100,
    "oz": 0,
    "fs": 10,
    "nx": 820,
    "ny": 820,
}

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

otherpar = {
    "lossless": False,
    "fresnel": True,
    "rms_height": 0.239 #0.6215
}

# first export basic single example stuff
params = pg.gen_params("REASON_VHF", "planetary_ice", domainpar, recpar, sourcepar, par=otherpar)
pg.export_params(params, f"VHF_params", directory="Simpar/.")

# --- MAKE SOURCE PATH ---

maxZ = 25e3 # maximum altitude of source path [m]
minZ = 1000e3 # minimum altitude of source path [m]

sz = pg.vert_source_path(params, minZ, maxZ, "source_path", direct="Simpar/.", log=True)

# --- RX OPENING WINDOW FILE ---

pg.track_Z_rxwin(sz, 250, "rx_window_positions", direct="Simpar/.")
