#!/bin/bash
for folderName in $(ls -1dv res-50um-additive-run-*/); do
	cd $folderName
	sed -n '142,$p' material.config > material_void.config
	cd ..
	echo "done $folderName"
done

