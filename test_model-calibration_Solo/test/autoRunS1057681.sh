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
	for i in 40; do # 2 4 8 10 16 20; do # 40; do # 80
		cd ${prefix}_${i}x${i}x${i};
		# # rm -f nohup.out;
		# # nohup mpirun -np $(cat numProcessors.dat) $DAMASK_spectral --load tension.load --geom single_phase_equiaxed_8x8x8.geom 2>&1 &
		# mpirun -np $(cat numProcessors.dat) $DAMASK_spectral --load tension.load --geom single_phase_equiaxed_${i}x${i}x${i}.geom
		# postResults single_phase_equiaxed_${i}x${i}x${i}_tension.spectralOut --cr f,p

		cd postProc/

		# filterTable < single_phase_equiaxed_${i}x${i}x${i}_tension.txt --white inc,1_f,1_p > stress_strain.log
		addStrainTensors single_phase_equiaxed_${i}x${i}x${i}_tension.txt --left --logarithmic
		addCauchy single_phase_equiaxed_${i}x${i}x${i}_tension.txt
		addMises single_phase_equiaxed_${i}x${i}x${i}_tension.txt --strain 'ln(V)' --stress Cauchy

		cd ..
		cd ..
	done
done
