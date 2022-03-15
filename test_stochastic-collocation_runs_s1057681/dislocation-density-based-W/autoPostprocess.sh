#!/bin/bash

rm -fv output.dat
for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName/

	cat output.dat >> ../output.dat
	cd postProc/
	# python3 ../../computeYieldStress.py
	cd ..

	echo "done $folderName"
	cd ..
done

