#!/bin/bash

for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName


	if [ -d "postProc" ]; then
		echo "postProc/ is available in $(basename $(pwd))"
		cd postProc
		python3 ../../computeYieldStress.py
		cd ..
	else
		echo "postProc/ is NOT available in $(basename $(pwd))"
	fi

	echo "done $folderName"
	cd ..
done

