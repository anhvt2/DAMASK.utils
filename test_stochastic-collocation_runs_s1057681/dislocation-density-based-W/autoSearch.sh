#!/bin/bash

for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName

	if [ -f "postProc/output.dat" ]; then
		# echo "postProc/ is available in $(basename $(pwd))"
		echo "done $folderName"
	else
		echo "postProc/output.dat not found in ${folderName}"
		echo "$(basename $(pwd))" >> ../submit.log
	fi


	cd ..
done

