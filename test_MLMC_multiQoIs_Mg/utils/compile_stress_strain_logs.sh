#!/bin/bash

for folderName in $(find . -name 'postProc'); do
	shortFolderName=$(echo $folderName | cut -d/ -f2)
	cp $shortFolderName/postProc/output.dat ./$shortFolderName-output.dat
	# echo $shortFolderName
done

