#!/bin/bash

for i in $(seq 10); do
	bash generateMsDream3d.sh
	for dimCell in $(cat dimCellList.dat); do
		folderName="${dimCell}x${dimCell}x${dimCell}"
		cd $folderName
		python3 ../../../plotms3d.py
		mv MgRve_${dimCell}x${dimCell}x${dimCell}.png ../MgRve_${dimCell}x${dimCell}x${dimCell}_sve-${i}.png
		cd ..
	done
done
