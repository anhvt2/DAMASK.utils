#!/bin/bash

### 50um attempt
export spkFileName="dump.additive_dogbone.2339"
export voidPc="3.000"

python3 $(which dump2npy) --dump ${spkFileName}
python3 $(which npy2geom) --npy ${spkFileName}.npy

python3 seedVoid.py \
    --origGeomFileName ${spkFileName}.geom \
    --voidPercentage ${voidPc} \
    --voidDictionary voidEquiaxed.geom \
    --phaseFileName phase_dump_12_out.npy

python3 padAirPolycrystals.py \
    --numAirVoxels=4 \
    --origGeomFileName="voidSeeded_${voidPc}pc_${spkFileName}.geom"

geom_check voidSeeded_${voidPc}pc_${spkFileName}.geom

geom_check padded_voidSeeded_${voidPc}pc_${spkFileName}.geom

python3 $(which geom2npy) --geom="padded_voidSeeded_${voidPc}pc_${spkFileName}.geom"

# python3 findGaugeLocations.py --geom="padded_voidSeeded_${voidPc}pc_${spkFileName}.npy" --resolution 50

python3 plotms3d_maskedDogbone.py --filename="padded_voidSeeded_${voidPc}pc_${spkFileName}.vtr"
python3 plotms3d_maskedDogbone.py --filename="padded_voidSeeded_${voidPc}pc_${spkFileName}.vtr" --nametag="voids"
python3 plotms3d_maskedDogbone.py --filename="padded_voidSeeded_${voidPc}pc_${spkFileName}.vtr" --nametag="solids"

cat material.config.preamble  | cat - material.config | sponge material.config

cp padded_voidSeeded_${voidPc}pc_${spkFileName}.geom main.geom
python3 $(which geom2npy) --geom main.geom
