#!/bin/bash

ii=0
maxSubmit=50

for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName

	if [ -d "postProc" ]; then
		echo "postProc/ is available in $(basename $(pwd))"
	else
		if [ "$ii" -lt "$maxSubmit" ]; then
			sdel
			ssubmit
			((ii++))
			echo "Re-submit job in $(basename $(pwd))"
		fi
	fi

	echo "done $folderName"
	cd ..
done

