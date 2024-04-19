
This folder was created to support a JCISE special issue on PINN for the anomaly detection project.

The SPPARKS/DAMASK simulations are adopted from `DAMASK.utils/test_CBayes_tensileDogbone/spk/res-50um-run-1`

# Instructions

1. SPPARKS visualization are done by converting `dump` files to figures with `pyvista`.

```shell
bash createMaskedMicrostructure.sh
```

The code to automate `createMaskedMicrostructure.sh` is as follows.

```shell
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
```

Upon which, 1, 10, 100, 1000 MCS-step microstructures are extracted. 

2. Perform ISML from `sklearn`:

This step creates a classification microstructure. As explained in `isml.py`: `1` means air, `2` means solid, `3` means detected anomaly. 

```shell
python3 isml.py
```

3. Create anomaly detection with masked microstructure on dogbone:

This step 
* (1) converts detected anomaly to a DAMASK geometry file `.geom`,
* (2) utilizes `geom_check` to create a `.vtr` file,
* (3) use `pyvista` to highlight the anomaly detection in dogbone

```shell
bash autoPostProcess.sh
```

