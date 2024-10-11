
# Seed voids to microstructure

Basic ideas in `test_CBayes_tensileDogbone/seedVoid/`:

1. Due to an array of SPPARKS simulations that share the same files, we seed voids for the microstructure here and move the geometry back to its original location **without** modifying the original file. 
1. The void percentage is an array of uniformly random, ordered percentages between [0,10], said 500 samples. 
1. Input files:
    * `voidEquiaxed.geom`: void dictionary from DREAM.3D
    * `phase_dump_12_out.npy`: phase file from CAD
    * `orientations.dat`: texture crystallography from DREAM.3D
    * `dump.additive_dogbone.2807`: microstructure from SPPARKS
1. Output files: 
    * `void+phase.*.npy`
    * `material.config`
    * `grainInfo.dat`
    * `httDB_${i}.geom`
    * `httDB_${i}.npy`
    * `httDB_${i}.vtr`
    * `seedVoid.log`
    * `gaugeFilter.txt`

# Implementation

##### Running PyVista on a headless server

Based on [https://github.com/pyvista/pyvista-support/issues/190](https://github.com/pyvista/pyvista-support/issues/190)

```shell
#!/bin/bash
set -x
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
set +x
exec "$@"
```

##### `run.sh`

```shell
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
    mv padded_voidSeeded_dump.additive_dogbone.2807.vti httDB_${i}.vti
    mv padded_voidSeeded_dump.additive_dogbone.2807.npy httDB_${i}.npy
    mv padded_voidSeeded_dump.additive_dogbone.2807.geom httDB_${i}.geom 
    mv padded_voidSeeded_dump.additive_dogbone.2807.vtr httDB_${i}.vtr
    mv padded_voidSeeded_dump.additive_dogbone.2807_voids.png httDB-voids-${i}.png 
    # Move output files into the original folder
    mv dump.additive_dogbone.2807 dump.additive_dogbone.2807.npy dump.additive_dogbone.2807.geom voidSeeded_dump.additive_dogbone.2807.npy voidSeeded_dump.additive_dogbone.2807.geom voidSeeded_dump.additive_dogbone.2807.vtr padded_voidSeeded_dump.additive_dogbone.2807.vti  material.config grainInfo.dat httDB_${i}.geom httDB_${i}.npy httDB_${i}.vtr void+phase-httDB_${i}.npy seedVoid.log gaugeFilter.txt ../spk/res-50um-additive-run-${i}/
    # Clean up
    rm ${spkFileName}
done
```
