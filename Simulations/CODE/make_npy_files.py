import os
import glob
import numpy as np
import argparse

# folder lists
#Facet_folders = ["F020", "F050", "F100", "F200"]
#Hurst_folders = ["H05"]
#RMSH_folders  = ["H000000", "H001152", "H002305", "H003457", "H004610"]
#ffolder = "F020"
#Hfolder = "H05"
#Rfolder = "H001152"

#roughness_model = "SimpleBackscatter"

# parse input
parser = argparse.ArgumentParser()
parser.add_argument("ffolder", type=str, help="Facet folder")
parser.add_argument("hfolder", type=str, help="Hurst exponent folder")
parser.add_argument("rfolder", type=str, help="RMS height folder")
parser.add_argument("roughness_model", type=str, help="Roughness model")
args = parser.parse_args()

ffolder = args.ffolder
hfolder = args.hfolder
rfolder = args.rfolder

model_dict = {
    "specular": "Lossless",
    "coherent_specular": "CoherentLoss",
    "simple_backscatter": "SimpleBackscatter"
}

roughness_model = model_dict[args.roughness_model]

# where to save the .npy files
NPY_DIR = f"Plots/{roughness_model}"
os.makedirs(NPY_DIR, exist_ok=True)

def load_datacube(ffolder, hfolder, rfolder):
    radar_paths = np.sort(
        glob.glob(f"Radargrams/{ffolder}/{hfolder}/{rfolder}/rdr*/rdrgrm.npy")
    )

    datacube = np.stack([np.load(p) for p in radar_paths], axis=-1)
    datacube = np.abs(datacube) ** 2
    return datacube

print(f"Loading in for {ffolder}/{hfolder}/{rfolder}")

cube = load_datacube(ffolder, hfolder, rfolder)

outname = f"{ffolder}{hfolder}{rfolder}.npy"
outpath = os.path.join(NPY_DIR, outname)
print(f"Saving to {outpath}")
np.save(outpath, cube)

