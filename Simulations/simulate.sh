#!/usr/bin/env bash
set -euo pipefail

# Set to true to run simulations even when the traces directory is populated.
# The simulation/compile scripts handle overwriting; this script does not delete files.
overwrite=false

Frs=(F020 F050 F100 F200)
Hrs=(H05)
rghs=(H000000 H001152 H002305 H003457 H004610)
instances=({00..31})

for Frs_value in "${Frs[@]}"; do
    for Hrs_value in "${Hrs[@]}"; do
        for rgh_value in "${rghs[@]}"; do
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
                    "$surface_dir/VHF_params.json" \
                    "$surface_dir/s00$i.fct" \
                    Simpar/target.txt \
                    "$traces_dir"

                python CODE/compileRdrgrm.py \
                    "$traces_dir" \
                    Simpar/VHF_params.pkl \
                    "$rdr_dir"

                printf '\n====== COMPLETED %s ======\n\n' "$i"
            done
        done
    done
done


#Frs="F020"
#Hrs="H05"
#rgh="H001152"

# iterate over number of instances for a given surface
#for i in {00..31};
#do
    
    # first make trace directory if it doesnt exist
#    if [ ! -d Radargrams/$Frs/$Hrs/$rgh/rdr$i/traces ]; then
#        mkdir -p Radargrams/$Frs/$Hrs/$rgh/rdr$i/traces;
#    fi

    # run the simulation code
#    ./radarSim "Surfaces/$Frs/$Hrs/$rgh/VHF_params.json" "Surfaces/$Frs/$Hrs/$rgh/s00$i.fct" Simpar/target.txt "Radargrams/$Frs/$Hrs/$rgh/rdr$i/traces"

    # compile output
#    python CODE/compileRdrgrm.py "Radargrams/$Frs/$Hrs/$rgh/rdr$i/traces" Simpar/VHF_params.pkl "Radargrams/$Frs/$Hrs/$rgh/rdr$i"
    
    # print
#    printf '\n====== COMPLETED %s ======\n\n' "$i"

#done
