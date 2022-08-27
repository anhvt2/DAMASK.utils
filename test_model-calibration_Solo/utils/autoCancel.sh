#!/bin/bash

modelName=$(cat modelName.dat)
numOfFolders=$(ls -1dv *_Iter*/ | wc -l)

for i in $(seq $numOfFolders); do
	cd ${modelName}_Iter${i}
	$cancel
	cd ..
done

