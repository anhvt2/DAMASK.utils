#!/bin/bash

for folderName in $(find . -name 'postProc'); do
	shortFolderName=$(echo $folderName | cut -d/ -f4)
	cp $folderName/output.dat ./$shortFolderName-output.dat
	cp $folderName/stress_strain.log ./$shortFolderName-stress_strain.log
	# echo $shortFolderName
done

