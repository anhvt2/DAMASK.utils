#!/bin/bash

for i in $(seq 0 28); do
    fileName="dump.${i}.out"

    # Create DAMASK/geom file
    python3 geom_spk2dmsk_RandomTexture.py -r 50 -d ${fileName} --phaseFileName='phase_dump_12_out.npy'

    # Create DAMASK/vtr file
    geomFileName="spk_dump_${i}_out.geom"
    geom_check ${geomFileName}

    # Create PNG file
    vtrFileName="spk_dump_${i}_out.vtr"
    python3 plotms3d_maskedDogbone.py --fileName ${vtrFileName}
    echo "Finished processing SPPARKS dump file: ${fileName}."
done
