#!/bin/bash

# Activate pyvista on headless server

numSimulations=2

for ((i=1; i<=${numSimulations}; i++)); do
    echo "Start simulation ${i}"

    # Generate microstructure
    bash generateMsDream3d.sh

    # Move file to local folder
    mkdir -p ${i}
    mv material.config simple2d.geom $i

    # # Run simulations
    cd ${i}
	cp ../tension.load .
	cp ../numProcessors.dat .
	cp ../sbatch.damask.srn .
	# ssubmit
    cd ..

    echo "End simulation $i"
done
