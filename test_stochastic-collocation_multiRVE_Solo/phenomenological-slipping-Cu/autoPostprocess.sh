#!/bin/bash

rm -fv output.dat
for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName/

	cd postProc/
	python3 ../../computeYieldStress.py
	cd ..
	cat output.dat >> ../output.dat

	echo "done $folderName"
	cd ..
done

