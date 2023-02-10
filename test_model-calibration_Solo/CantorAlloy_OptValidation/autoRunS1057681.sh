#!/bin/bash

# for j in $(seq 2 10); do
# 	bash generateMsDream3d.sh
# 	mv 2x2x2 sve${j}_2x2x2
# 	mv 4x4x4 sve${j}_4x4x4
# 	mv 8x8x8 sve${j}_8x8x8
# 	mv 10x10x10 sve${j}_10x10x10
# 	mv 16x16x16 sve${j}_16x16x16
# 	mv 20x20x20 sve${j}_20x20x20
# 	mv 40x40x40 sve${j}_40x40x40
# 	mv 80x80x80 sve${j}_80x80x80
# done


# how to run: nohup bash autoRunS1057681.sh 2>&1 &

for j in $(seq 10); do # 1 2
	prefix="sve${j}"
	for i in 2 4 8 10 16; do # 20; do # 40; do # 80
		cd ${prefix}_${i}x${i}x${i};
		python3 ../../../plotms3d.py
		cp single_phase_equiaxed_${i}x${i}x${i}.png ../single_phase_equiaxed_${i}x${i}x${i}_${prefix}_${i}x${i}x${i}.png

		mpirun -np $(cat numProcessors.dat) $DAMASK_spectral --load tension.load --geom single_phase_equiaxed_${i}x${i}x${i}.geom
		postResults single_phase_equiaxed_${i}x${i}x${i}_tension.spectralOut --cr f,p

		cd postProc/

		addStrainTensors single_phase_equiaxed_${i}x${i}x${i}_tension.txt --left --logarithmic
		addCauchy single_phase_equiaxed_${i}x${i}x${i}_tension.txt
		addMises single_phase_equiaxed_${i}x${i}x${i}_tension.txt --strain 'ln(V)' --stress Cauchy
		filterTable < single_phase_equiaxed_${i}x${i}x${i}_tension.txt --white inc,'1_ln(V)','1_Cauchy' > stress_strain.log

		cd ..
		cd ..
		python3 computeLossFunction.py --f=sve${j}_${i}x${i}x${i}/ -p 1
	done
done
