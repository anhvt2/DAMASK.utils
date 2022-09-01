#!/bin/bash


modelName="mlmcSS304L"

rm -rfv ${modelName}_Iter{3,4,5,6,7,8,9}
rm -rfv ${modelName}_Iter??
rm -rfv ${modelName}_Iter???

for i in $(seq 3 20); do
	folderName="${modelName}_Iter${i}"
	cp -rfv ${modelName}_Template/ ${folderName}
	
	cd $folderName
	python3 random6d.py # create input.py
	python3 parse2MaterialConfig.py
	cd ..
	
	echo "done $folderName"; echo; echo;
done

