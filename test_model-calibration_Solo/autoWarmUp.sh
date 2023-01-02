#!/bin/bash

modelName="MonteCarloCantorAlloy"

for i in $(seq 1 9); do
	folderName="${modelName}_Iter${i}"

	cd $folderName
	cp ../${modelName}_Template/reparse2MaterialConfig.py .
	python3 reparse2MaterialConfig.py
	cd ..
	echo "done $folderName"; echo; echo;
done

