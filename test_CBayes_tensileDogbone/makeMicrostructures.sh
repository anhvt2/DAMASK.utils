#!/bin/bash

python3 seedVoid.py \
    --origGeomFileName spk_dump_12_out.geom \
    --percentage 1.5 \
    --phaseFileName phase_dump_12_out.npy
# out: voidSeeded_1.500pc_spk_dump_12_out.geom

python3 padAirPolycrystals.py \
    --origGeomFileName voidSeeded_1.500pc_spk_dump_12_out.geom
# out: padded_voidSeeded_1.500pc_spk_dump_12_out.geom


cat ./material.config.preamble  | cat - ./material.config | sponge ./material.config
