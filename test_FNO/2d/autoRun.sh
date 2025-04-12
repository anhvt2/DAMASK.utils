#!/bin/bash

# Activate pyvista on headless server

numSimulations=2

for ((i=1; i<=${numSimulations}; i++)); do
    echo "Start simulation ${i}"

    # Generate microstructure
    bash generateMsDream3d.sh

    # Move file to local folder
    mkdir -p ${i}
    mv dream3d.material.config material.config simple2d.geom $i

    # Run simulations
    cd ${i}
    cp ../tension.load .
    geom_check simple2d.geom
    python3 ../../../geom2png.py --geom simple2d.geom
    $DAMASK_spectral --load tension.load --geom simple2d.geom
    postResults *.spectralOut --cr f,p

    cd postProc/
    addStrainTensors simple2d_tension.txt --left --logarithmic
    addCauchy simple2d_tension.txt
    addMises simple2d_tension.txt --strain 'ln(V)' --stress Cauchy
    filterTable < simple2d_tension.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > stress_strain.log
    cd ..
    postResults \
        --cr fp,f,p,grainrotation,texture \
        --split \
        --separation x,y,z \
        --increments \
        --range 1 30 1 simple2d_tension.spectralOut 2>&1 > log.postResults
    cd ..

    echo "End simulation $i"
done
