#!/bin/bash

# for i in $(seq -f "%03g" 1 100); do
for i in $(seq 500); do
# for i in $(cat tmp2.txt); do
	folderName="res-50um-additive-run-${i}"
	templateFolderName="res-50um-additive"
	cd $folderName

    rm sbatch.*
    ln -sf httDB_${i}.geom main.geom
    cp ../damask-test/gaugeFilter.txt .
    cp ../damask-test/numProcessors.dat .
    cp ../damask-test/tension.load .
    cp ../damask-test/numerics.config .
    cp ../damask-test/sbatch.damask.srn .
	
    ssubmit
	cd ..
done

