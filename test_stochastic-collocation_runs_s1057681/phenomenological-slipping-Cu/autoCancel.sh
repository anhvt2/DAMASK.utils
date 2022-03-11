#!/bin/bash

for folderName in $(ls -1dv sg_input_*/); do
# for folderName in $(cat submit.log); do
	cd $folderName

	sdel

	echo "done $folderName"
	cd ..
done

