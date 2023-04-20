#!/bin/bash

for folderName in $(find . -name 'postProc'); do
	shortFolderName=$(echo $folderName | cut -d/ -f2)
	cp $shortFolderName/postProc/output.dat ./$shortFolderName-output.dat
	cp $shortFolderName/postProc/stress_strain.log ./$shortFolderName-stress_strain.log
	# echo $shortFolderName
done

