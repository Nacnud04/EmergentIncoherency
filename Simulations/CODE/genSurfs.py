import argparse
import numpy as np

from roughSynth import fractal
from roughSynth import export_grid_to_facet

# parse input
parser = argparse.ArgumentParser()
parser.add_argument("H", type=float, help="Hurst exponent")
parser.add_argument("N", type=int, help="Samples per side")
parser.add_argument("L", type=float, help="Side length of surface")
parser.add_argument("Baseline", type=float, help="Baseline/PointToPoint/Profile length")
parser.add_argument("OutputDir", type=str, help="Output directory")
parser.add_argument("--RMSHeight", type=float, help="RMS height over profile/baseline length")
parser.add_argument("--RMSDeviation", type=float, help="RMS deviation over point to point spacing")
parser.add_argument("--N_surfs", type=int, help="Number of surfaces to generation", default=2)
parser.add_argument("--seed", type=int, help="Seed")

args = parser.parse_args()

# update N_Surfs to multiple of two
if args.N_surfs % 2 == 1:
    print(f"Provided N_surfs was odd. Increasing by 1 to make even.")
    args.N_surfs += 1
N_iters = int(args.N_surfs / 2)

# make sure either an RMS Height or RMS Deviation is provided
if args.RMSHeight is None and args.RMSDeviation is None:
    raise ValueError("Either RMSHeight or RMSDeviation must be provided as an input!")

# find R
if 2 * args.H <= 1.5:
    R = 1.0
else:
    R = 2.0

# === GENERATE SURFACES === 

for i in range(N_iters):

    if args.seed:
        seed = args.seed + i
    else:
        seed = None

    # first produce surface
    if args.RMSHeight is not None:
        f1, f2, xs, ys = fractal(args.N, args.N, R, args.H, args.L, eps=args.RMSHeight, baseline=args.Baseline, seed=seed)
    else:
        raise NotImplementedError("RMS devation / RMS slope approach not fully implemented in roughSynth")

    # then export surface into directory
    xxs, yys = np.meshgrid(xs - args.L / 2, ys - args.L / 2)
    export_grid_to_facet(xxs, yys, f1, f"{args.OutputDir}/s{(i*2):04d}.fct")
    export_grid_to_facet(xxs, yys, f2, f"{args.OutputDir}/s{(i*2)+1:04d}.fct")
