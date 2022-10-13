#!/bin/bash

modelName="SS304L"

for i in $(seq 1 9); do
# for i in $(seq 2 10); do
	folderName="${modelName}_Iter${i}"

	cd $folderName
	# ssubmit
	# python3 ../test.py
	echo >> input.dat
	cd ..
	echo "done $folderName"; echo; echo;
done

