#!/bin/bash

# how to run: nohup bash autoRunS1057681.sh 2>&1 &

for j in 3 4 5; do # 1 2
	prefix="sve${j}"
	for i in 2 4 8 10 16 20 40; do # 80
		cd ${prefix}_${i}x${i}x${i};
		# rm -f nohup.out;
		# nohup mpirun -np $(cat numProcessors.dat) $DAMASK_spectral --load tension.load --geom single_phase_equiaxed_8x8x8.geom 2>&1 &
		mpirun -np $(cat numProcessors.dat) $DAMASK_spectral --load tension.load --geom single_phase_equiaxed_${i}x${i}x${i}.geom
		postResults single_phase_equiaxed_${i}x${i}x${i}_tension.spectralOut --cr f,p
		cd postProc/
		filterTable < single_phase_equiaxed_${i}x${i}x${i}_tension.txt --white inc,1_f,1_p > stress_strain.log
		cd ..
		cd ..
	done
done
