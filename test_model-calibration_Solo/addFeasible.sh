#!/bin/bash
for folderName in $(ls -1dv *_Iter*/); do
	cd $folderName
	if [ ! -e "feasible.dat" ]; then
		echo "$folderName does not contain feasible.dat"
		echo "0" > feasible.dat
	fi
	cd ..
done

