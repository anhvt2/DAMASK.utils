#!/bin/bash

python3 ../../geom_spk2dmsk.py -r 50 -d 'dump.12.out' --phaseFileName='../phase_dump_12_out.npy'

for pc in 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00; do
    rm -rfv ${pc}
    python3 ../../seedVoid.py \
        --origGeomFileName spk_dump_12_out.geom \
        --percentage ${pc} \
        --phaseFileName phase_dump_12_out.npy
    # out: voidSeeded_1.500pc_spk_dump_12_out.geom

    python3 ../../padAirPolycrystals.py \
        --origGeomFileName voidSeeded_${pc}0pc_spk_dump_12_out.geom
        # --origGeomFileName voidSeeded_1.500pc_spk_dump_12_out.geom
    # out: padded_voidSeeded_1.500pc_spk_dump_12_out.geom

    mkdir ${pc}
    mv material.config seedVoid.log ${pc}
    cd ${pc}
    cat ../../../material.config.preamble  | cat - ./material.config | sponge ./material.config
    cd ..
    echo "done ${pc} percentage of voids"
done


