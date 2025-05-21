#!/bin/bash

### 50um attempt
export spkFileName="dump.additive_dogbone.2339"
export voidPc="3.000"

python3 dump2npy.py --dump ${spkFileName}
python3 npy2geom.py --npy ${spkFileName}.npy

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

python3 geom2npy.py --geom="padded_voidSeeded_${voidPc}pc_${spkFileName}.geom"

# python3 findGaugeLocations.py --geom="padded_voidSeeded_${voidPc}pc_${spkFileName}.npy" --resolution 50

python3 plotms3d_maskedDogbone.py --fileName="padded_voidSeeded_${voidPc}pc_${spkFileName}.vtr"
python3 plotms3d_maskedDogbone.py --fileName="padded_voidSeeded_${voidPc}pc_${spkFileName}.vtr" --name_tag = "voids"
python3 plotms3d_maskedDogbone.py --fileName="padded_voidSeeded_${voidPc}pc_${spkFileName}.vtr" --name_tag = "solids"

cat material.config.preamble  | cat - material.config | sponge material.config

cp padded_voidSeeded_${voidPc}pc_${spkFileName}.geom main.geom
