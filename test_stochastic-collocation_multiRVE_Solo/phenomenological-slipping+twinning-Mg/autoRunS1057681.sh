#!/bin/bash

# for folderName in $(ls -1dv sg_input_*/); do
for folderName in $(cat submit.log); do
	cd $folderName

# 	if [ -d "postProc" ]; then
# 		echo "postProc/ is available in $(basename $(pwd))"
# 	else
		echo "Running $(basename $(pwd)) on s1057681..."
		ln -sf ../run_damask.sh .
		echo "16" > numProcessors.dat
		bash run_damask.sh
# 	fi

	echo "done $folderName"
	cd ..
done

