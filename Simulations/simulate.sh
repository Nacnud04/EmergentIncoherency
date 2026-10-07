#!/usr/bin/env bash

rgh="H001256"

# iterate over number of instances for a given surface
for i in {00..31};
do
    
    # first make trace directory if it doesnt exist
    if [ ! -d Radargrams/H40/$rgh/rdr$i/traces ]; then
        mkdir -p Radargrams/H40/$rgh/rdr$i/traces;
    fi

    # run the simulation code
    ./radarSim Simpar/VHF_params.json "Surfaces/H40/$rgh/s00$i.fct" Simpar/target.txt "Radargrams/H40/$rgh/rdr$i/traces"

    # compile output
    python CODE/compileRdrgrm.py "Radargrams/H40/$rgh/rdr$i/traces" Simpar/VHF_params.pkl "Radargrams/H40/$rgh/rdr$i"
    
    # print
    printf '\n====== COMPLETED %s ======\n\n' "$i"

done
