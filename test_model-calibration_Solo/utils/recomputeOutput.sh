#!/bin/bash

# use this script if you would like to change getOutput2.py for testing purposes

modelName=$(cat modelName.dat)
for folder in $(ls -1dv ${modelName}_Iter*/); do
	cd $folder
	python3 ../getOutput2.py
	echo "done $folder"
	cd ..
done
