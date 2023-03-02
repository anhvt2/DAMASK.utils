#!/bin/bash

for fN in $(ls -1dv *sve*/); do
	cd $fN

	if [ ! -e "postProc/youngModulus.out"  ]; then
#		rm -rfv single_phase_equiaxed_tension* postProc/
		rm -rfv postProc/
#		cp ../sbatch.damask.solo .
#		cp ../tension.load .
#		cp ../computeYoungModulus.py .
#		rm -fv sbatch.damask2.cades
		ln -sf ../sbatch.damask.solo
		sdel
		ssubmit
		echo "resetting $fN"
	fi

	cd ..
	echo "done $fN"
done
