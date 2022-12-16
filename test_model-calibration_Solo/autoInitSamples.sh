#!/bin/bash

modelName="MonteCarloCantorAlloy"

rm -rfv ${modelName}_Iter{1,2,3,4,5,6,7,8,9}
rm -rfv ${modelName}_Iter??
rm -rfv ${modelName}_Iter???

for i in $(seq 1 9); do
	folderName="${modelName}_Iter${i}"
	cp -rfv ${modelName}_Template/ ${folderName}

	cd $folderName
	python3 randomSampleInput.py # create input.dat
	echo >> input.dat
	python3 parse2MaterialConfig.py
	echo 0 > batchID.dat
	echo 1 > feasible.dat
	echo 0 > acquisitionScheme.dat
	echo 1 > complete.dat
	cd ..

	echo "done $folderName"; echo; echo;
done

