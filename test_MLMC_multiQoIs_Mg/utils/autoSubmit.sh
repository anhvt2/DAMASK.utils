#!/bin/bash

for folderName in $(ls -1dv hpc_level*sample*/); do
	cd $folderName
	ssubmit
	cd ..
	echo "done $folderName"
done
