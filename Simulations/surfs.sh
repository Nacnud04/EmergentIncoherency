# make some smoother surfaces
#!/usr/bin/env bash
set -euo pipefail

force=0
if [[ "${1:-}" == "--force" ]]; then
    force=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--force]" >&2
    exit 2
fi

H_values=(0.05)

# Matching entries at each index describe one facet configuration.
N_values=(3000 1640 820 410)
L_values=(6000 8200 8200 8200)
facet_tags=(020 050 100 200)

RMSHeight_values=(0.0000 0.1152 0.2305 0.3457 0.4610)
Baseline_values=(10000 1 1 1 1)
height_tags=(000000 001152 002305 003457 004610)

for H in "${H_values[@]}"; do
    for i in "${!N_values[@]}"; do
        N="${N_values[$i]}"
        L="${L_values[$i]}"

        for j in "${!RMSHeight_values[@]}"; do
            Baseline="${Baseline_values[$j]}"
            RMSHeight="${RMSHeight_values[$j]}"
            OutputDir="Surfaces/F${facet_tags[$i]}/H05/H${height_tags[$j]}"

            mkdir -p "$OutputDir"

            if (( ! force )) &&
                [[ -n "$(find "$OutputDir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
                echo "Skipping populated directory: $OutputDir"
                continue
            fi

            python CODE/genSurfs.py \
                "$H" "$N" "$L" "$Baseline" "$OutputDir" \
                --RMSHeight "$RMSHeight" \
                --N_surfs 32 \
                --seed 2026
        done
    done
done


# 10 - meter facet size (2 lambda)
# generate surfaces similar in structure to floating antarctic ice
#python CODE/genSurfs.py 0.4 820 8200 10000 Surfaces/F100/H40/H001256 --RMSHeight 5.0 --N_surfs 32 --seed 2026

# generate surfaces similar in structure to grounded antarctic ice
#python CODE/genSurfs.py 0.4 820 8200 10000 Surfaces/F100/H40/H003265 --RMSHeight 13.0 --N_surfs 32 --seed 2026
