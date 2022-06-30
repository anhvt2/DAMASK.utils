#!/bin/bash

# postproc. prefix to distinguish between 
# 	on-the-fly and post-prcess
rm -v postproc.{F,S,Y,folderName,acquisitionScheme}.dat 

modelName=$(cat modelName.dat)
for folderName in $(ls -1dv ${modelName}*/); do
	tr '\n' ',' < $folderName/input.dat >> postproc.S.dat
	echo >> postproc.S.dat
	
	if [ -f $folderName/feasible.dat ] && [ -f $folderName/output.dat ]; then
		cat $folderName/feasible.dat >> postproc.F.dat
		cat $folderName/output.dat >> postproc.Y.dat
	else
		echo 0 >> postproc.F.dat
		echo 0 >> postproc.Y.dat
	fi

	cat $folderName/acquisitionScheme.dat >> postproc.acquisitionScheme.dat	

	echo "$folderName" >> postproc.folderName.dat
	echo "done $folderName"
done
