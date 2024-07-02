#!/bin/bash

python3 seedVoid.py     --origGeomFileName potts-12_3d.975.geom     --voidPercentage 3     --voidDictionary voidEquiaxed.geom     --phaseFileName phase_dump_12_out.npy

geom_check voidSeeded_3.000pc_potts-12_3d.975.geom
python3 plotms3d_maskedDogbone.py --fileName='voidSeeded_3.000pc_potts-12_3d.975.vtr'
