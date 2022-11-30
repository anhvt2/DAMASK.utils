#!/bin/bash

for d in $(cat dimCellList.dat); do
	folderName="${d}x${d}x${d}"
	cd $folderName
	python3 ../../plotms3d.py
	mv single_phase_equiaxed_${folderName}.png single_phase_equiaxed_${folderName}_oceancmap.png
	cd ..
done
