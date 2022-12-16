#!/bin/bash

modelName="MonteCarloCantorAlloy"

for i in $(seq 1 9); do
	folderName="${modelName}_Iter${i}"

	cd $folderName
	ssubmit
	cd ..
	echo "done $folderName"; echo; echo;
done

