#!/bin/bash

# for i in $(seq 20); do
for i in $(seq 10); do
	folderName="SS304L_Iter${i}"
	cd $folderName
	# cp ../mlmcSS304L_Template/sbatch.damask.solo .
	cp ../SS304L_Template/computeLossFunction.py . 
	python3 computeLossFunction.py --f=8x8x8/ -p 0
	echo 1 > complete.dat
	echo 1 > feasible.dat
	echo 0 > batchID.dat
	echo 0 > acquisitionScheme.dat
	cp *_8x8x8.png ..
	cd ..
	echo "done $folderName"; echo; echo;
done
