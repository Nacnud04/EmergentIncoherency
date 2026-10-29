#!/usr/bin/env bash
set -euo pipefail

# Set to true to run simulations even when the traces directory is populated.
# The simulation/compile scripts handle overwriting; this script does not delete files.
overwrite=true

Frs=(F020 F050 F100 F200)
Hrs=(H10)
rghs=(H001152 H002305 H003457 H004610)
models=("specular" "coherent_specular" "simple_backscatter")
instances=({00..31})

for Frs_value in "${Frs[@]}"; do
    for Hrs_value in "${Hrs[@]}"; do
        for rgh_value in "${rghs[@]}"; do
            for model in "${models[@]}"; do
                for i in "${instances[@]}"; do
                    surface_dir="Surfaces/$Frs_value/$Hrs_value/$rgh_value"
                    rdr_dir="Radargrams/$Frs_value/$Hrs_value/$rgh_value/rdr$i"
                    traces_dir="$rdr_dir/traces"

                    # Skip this instance if traces already contains files or directories,
                    # unless overwrite is enabled.
                    if [[ "$overwrite" != true && -d "$traces_dir" ]] &&
                    [[ -n "$(find "$traces_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
                        echo "Skipping populated traces directory: $traces_dir"
                        continue
                    fi

                    mkdir -p "$traces_dir"

                    ./radarSim \
                        "$surface_dir/VHF_$model.json" \
                        "$surface_dir/s00$i.fct" \
                        Simpar/target.txt \
                        "$traces_dir"

                    python CODE/compileRdrgrm.py \
                        "$traces_dir" \
                        Simpar/VHF_params.pkl \
                        "$rdr_dir"

                    printf '\n====== COMPLETED %s ======\n\n' "$i"
                done
                # at the end of the loop make and export the .npy file
                python CODE/make_npy_files.py \
                    "$Frs_value" \
                    "$Hrs_value" \
                    "$rgh_value" \
                    "$model" 
            done
        done
    done
done

