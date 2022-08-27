#!/bin/bash

numInitPoint=3 # change this according to the actual number of folder -- DO NOT AUTOMATE due to the risk of override information in warm-restart
modelName=$(cat modelName.dat)
numOfFolders=$(ls -1dv ${modelName}_Iter*/ | wc -l)

for i in $(seq ${numOfFolders}); do
	cd ${modelName}_Iter${i}

	echo 0 > batchID.dat
	echo 0 > acquisitionScheme.dat

	cd ..
	echo "done ${modelName}_Iter${i}"
done


