#!/bin/bash


for j in $(seq 10); do # 1 2
	prefix="sve${j}"
	for i in 2 4 8 10 16; do # 20; do # 40; do # 80
		cd ${prefix}_${i}x${i}x${i};
		geom_check single_phase_equiaxed_${i}x${i}x${i}.geom
		python3 ../../../plotms3d.py
		cp single_phase_equiaxed_${i}x${i}x${i}.png ../single_phase_equiaxed_${i}x${i}x${i}_${prefix}.png

		cd ..

	done
done
