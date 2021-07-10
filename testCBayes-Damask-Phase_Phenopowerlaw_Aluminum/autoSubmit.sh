#!/bin/bash

for fN in $(ls -1dv *sve*/); do
	cd $fN

	if [ ! -e "postProc/youngModulus.out"  ]; then
		rm -rfv single_phase_equiaxed_tension* postProc/
		cp ../sbatch.damask.solo .
		cp ../tension.load .
		cp ../computeYoungModulus.py .
		sdel
		ssubmit
		echo "resetting $fN"
	fi

	cd ..
	echo "done $fN"
done
