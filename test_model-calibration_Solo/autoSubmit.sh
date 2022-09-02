#!/bin/bash

modelName="mlmcSS304L"

for i in $(seq 1 20); do
	folderName="${modelName}_Iter${i}"

	cd $folderName
	ssubmit
	cd ..
	echo "done $folderName"; echo; echo;
done

