#!/bin/bash

export spkFileName="dump.additive_dogbone.2807"
for i in $(seq 1); do
    cp ../spk/res-50um-additive-run-${i}/${spkFileName} . 
    python3 ../../dump2npy.py --dump ${spkFileName}
    python3 ../../npy2geom.py --npy ${spkFileName}.npy
    # Seed voids
    python3 seedVoid.py \
        --origGeomFileName ${spkFileName}.geom \
        --voidPercentage $(sed -n ${i}p porosity.txt) \
        --voidDictionary voidEquiaxed.geom \
        --phaseFileName phase_dump_12_out.npy
    # Pad air
    python3 padAirPolycrystals.py --origGeomFileName="voidSeeded_${spkFileName}.geom"
    # Check geometries
    geom_check voidSeeded_${spkFileName}.geom
    geom_check padded_voidSeeded_${spkFileName}.geom
    # Convert to numpy
    python3 geom2npy.py --geom="padded_voidSeeded_${spkFileName}.geom"
    # Make gaugeFilter.txt for DAMASK post-processing homogenization
    python3 findGaugeLocations.py --geom="padded_voidSeeded_${spkFileName}.npy" --resolution 50
    # Visualize voids for .png
    python3 plotms3d_maskedDogbone.py --fileName="padded_voidSeeded_${spkFileName}.vtr" --nameTag 'voids'
    python3 plotms3d_maskedDogbone.py --fileName="padded_voidSeeded_${spkFileName}.vtr" --nameTag 'solids'
    # Make material.config for DAMASK
    cat material.config.preamble  | cat - material.config | sponge material.config
    # Rename
    mv void+phase_dump_12_out.npy void+phase-httDB_${i}.npy
    mv padded_voidSeeded_${spkFileName}.vti httDB_${i}.vti
    mv padded_voidSeeded_${spkFileName}.npy httDB_${i}.npy
    mv padded_voidSeeded_${spkFileName}.geom httDB_${i}.geom 
    mv padded_voidSeeded_${spkFileName}.vtr httDB_${i}.vtr
    mv padded_voidSeeded_${spkFileName}_voids.png httDB-voids-${i}.png 
    # Move output files (except for the original SPPARKS dump file) into the original folder
    mv ${spkFileName}.npy ${spkFileName}.geom voidSeeded_${spkFileName}.npy voidSeeded_${spkFileName}.geom voidSeeded_${spkFileName}.vtr material.config grainInfo.dat httDB_${i}.geom httDB_${i}.vti httDB_${i}.npy httDB_${i}.vtr httDB-voids-${i}.png void+phase-httDB_${i}.npy seedVoid.log gaugeFilter.txt ../spk/res-50um-additive-run-${i}/
    # Clean up
    rm ${spkFileName}
done
