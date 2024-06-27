
# POD ROM for oligocrystal with different strain rates/temperature
```shell
# Run SPPARKS
cd spk/
mpirun -np 8 spk < in.potts_3d
cd ..

# Run DREAM.3D for generating a dictionary of void morphology
cd dream3d-void-libs/

cd ../

# Run seedVoid.py
cd seedVoid/
python3 seedVoid.py \
    --origGeomFileName spk_dump_12_out.geom \
    --percentage 1 \
    --voidDictionary voidEquiaxed.geom \
    --phaseFileName phase_dump_12_out.npy
cd ..
```

## SPPARKS

Run `in.potts_3d` with hybrid phase-field/kMC with sufficiently long trajectory.

## DREAM.3D

Sample orientation, and voids morphology.

## seedVoid.py

Sample void throughout the computational domain

## DAMASK

Make sure to use dislocation-density-based constitutive model for high-/low-strain rate with different temperature

# ROM

# 3D Visualization
