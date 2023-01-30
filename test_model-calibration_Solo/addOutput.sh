#!/bin/bash
for folderName in $(ls -1dv *_Iter*/); do
	cd $folderName
	if [ ! -e "output.dat" ]; then
		echo "$folderName does not contain output.dat"
		echo "-1.00000000e+02" > output.dat
	fi
	cd ..
done

