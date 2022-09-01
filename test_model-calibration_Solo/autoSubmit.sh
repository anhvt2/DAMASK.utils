#!/bin/bash

for i in $(seq 3 20); do
	folderName="${modelName}_Iter${i}"

	cd $folderName
	ssubmit
	cd ..
	echo "done $folderName"; echo; echo;
done

