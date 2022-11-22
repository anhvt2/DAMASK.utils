#!/bin/bash
for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName

	rm -f tension.load
	rm -f sbatch.damask.solo
	rm -f run_damask.sh
	rm -f numProcessors.dat
	rm -f numerics.config
	rm -f output.dat
	rm -f material.config

	cd ..
	echo "done $folderName"
done

