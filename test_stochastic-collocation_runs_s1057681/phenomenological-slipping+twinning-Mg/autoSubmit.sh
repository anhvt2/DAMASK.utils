#!/bin/bash

# for folderName in $(ls -1dv sg_input_*/); do
for folderName in $(cat submit.log); do
	cd $folderName

#	if [ -d "postProc" ]; then
#		echo "postProc/ is available in $(basename $(pwd))"
#	else
		sdel; rm -rfv postProc/;
		ssubmit
		echo "Re-submit job in $(basename $(pwd))"
#		echo "$(basename $(pwd))" >> ../submit.log
#	fi

	echo "done $folderName"
	cd ..
done

