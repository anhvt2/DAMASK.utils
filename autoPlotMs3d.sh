#!/bin/bash

for folderName in $(ls -1dv *x*x*/); do
	cd $folderName

	python3 ../../../plotms3d.py --nameTag="${folderName}"

	cd ..
	echo "done $folderName"
done
