#!/bin/bash

numOfMs=5 # number of microstructures

for mu in 3.50 3.25 3.00 2.75 2.50 2.25 2.00 1.75 1.50 1.25; do
    for sigma in 0.35 0.30 0.25 0.20 0.15 0.10 ; do
		# for i in $(seq ${numOfMs}); do
		for i in $(seq 6 30); do
			dimCell=64;
			folderName="${dimCell}x${dimCell}x${dimCell}-mu-${mu}-sigma-${sigma}-sve-${i}"
			cp -rfv template/ ${folderName}
			echo ${mu} > ${folderName}/mu.dat
			echo ${sigma} > ${folderName}/sigma.dat
			echo ${i} > ${folderName}/msId.dat
		done
	done
done

