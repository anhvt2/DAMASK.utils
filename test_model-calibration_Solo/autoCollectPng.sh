#!/bin/bash

modelName="MonteCarloCantorAlloy"

# for i in $(seq 1 9); do
#	folderName="${modelName}_Iter${i}"
for folderName in $(ls -1dv ${modelName}_Iter*/); do

	cd $folderName
	cp -v compareExpComp_${modelName}_*png ..
	cd ..
	echo "done $folderName"; echo; echo;
done

