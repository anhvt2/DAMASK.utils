#!/bin/bash

### 50um attempt
export spkFileName="dump.additive_dogbone.2807"
export voidPc="3.000"

python3 seedVoid.py \
    --origGeomFileName ${spkFileName}.geom \
    --voidPercentage ${voidPc} \
    --voidDictionary voidEquiaxed.geom \
    --phaseFileName phase_dump_12_out.npy

### 10 um attempt
# export spkFileName="additive_dogbone.1817"
# python3 seedVoid.py \
#     --origGeomFileName ${spkFileName}.geom \
#     --voidPercentage 3 \
#     --voidDictionary voidEquiaxed.geom \
#     --phaseFileName phase_dump_10_out.npy

python3 padAirPolycrystals.py --origGeomFileName="voidSeeded_${voidPc}pc_${spkFileName}.geom"

geom_check voidSeeded_${voidPc}pc_${spkFileName}.geom

geom_check padded_voidSeeded_${voidPc}pc_${spkFileName}.geom

python3 geom2npy.py --geom="padded_voidSeeded_${voidPc}pc_${spkFileName}.geom"

python3 findGaugeLocations.py --geom="padded_voidSeeded_${voidPc}pc_${spkFileName}.npy" --resolution 50

python3 plotms3d_maskedDogbone.py --fileName="padded_voidSeeded_${voidPc}pc_${spkFileName}.vtr"

cat material.config.preamble  | cat - material.config | sponge material.config

cp padded_voidSeeded_${voidPc}pc_${spkFileName}.geom ../damask/main.geom
