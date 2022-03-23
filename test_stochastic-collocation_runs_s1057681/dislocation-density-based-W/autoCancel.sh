#!/bin/bash

for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName

	sdel

	echo "done $folderName"
	cd ..
done

