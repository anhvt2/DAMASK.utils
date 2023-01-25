#!/bin/bash

for folderName in $(ls -1dv *_Iter*/); do
	cd $folderName
	sdel
	cd ..
	echo "done $folderName"
done
