#!/bin/bash

numOfMs=100 # number of microstructures

for mu in 3.00 2.75 2.50 2.25 2.00 1.75; do
    for sigma in 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05; do
		for i in $(seq ${numOfMs}); do
			folderName="64x64x64-mu-${mu}-sigma-${sigma}-sve-${i}"
			cp -rfv template/ ${folderName}
			echo ${mu} > ${folderName}/mu.dat
			echo ${sigma} > ${folderName}/sigma.dat
			echo ${i} > ${folderName}/msId.dat
		done
	done
done

