import numpy as np
import glob
import argparse
import pickle

# parse input
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to directory with traces")
parser.add_argument("param", type=str, help="Parameter file")
parser.add_argument("output", type=str, help="Output directory")
args = parser.parse_args()

with open(args.param, 'rb') as hdl:
    par = pickle.load(hdl)


filenames = glob.glob(f"{args.path}/s*.txt")

if len(filenames) == 0:
    raise ValueError(f"No files found in {args.path} with pattern s*.txt")

# sort filenames to ensure correct order
filenames.sort()

rdrgrm = []
for i, f in enumerate(filenames):
    if i < par['ns']:
        index = int(f.split("/")[-1][1:-4])
        print(f"Compiling radargram... {index+1}/{par['ns']}", end="           \r")
        arr = np.loadtxt(f).T
        col = arr[0] + 1j * arr[1]
        rdrgrm.append(col)

print("",end='\n')

rdrgrm = np.array(rdrgrm).T
np.save(f"{args.output}/rdrgrm.npy", rdrgrm)

import matplotlib.pyplot as plt

img = 20*np.log10(np.abs(rdrgrm))
plt.imsave(f"{args.output}/rdrgrm.png", img, cmap="viridis")
