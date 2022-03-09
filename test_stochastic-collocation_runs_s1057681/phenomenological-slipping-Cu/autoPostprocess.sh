#!/bin/bash

for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName/postProc/

	python3 ../../computeYieldStress.py

	echo "done $folderName"
	cd ../../
done

