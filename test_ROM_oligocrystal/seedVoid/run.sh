#!/bin/bash

export spkFileName="potts-12_3d.975"
python3 seedVoid.py     --origGeomFileName ${spkFileName}.geom     --voidPercentage 3     --voidDictionary voidEquiaxed.geom     --phaseFileName phase_dump_12_out.npy

geom_check voidSeeded_3.000pc_${spkFileName}.geom
python3 plotms3d_maskedDogbone.py --fileName="voidSeeded_3.000pc_${spkFileName}.vtr"
cat material.config.preamble  | cat - material.config | sponge material.config
