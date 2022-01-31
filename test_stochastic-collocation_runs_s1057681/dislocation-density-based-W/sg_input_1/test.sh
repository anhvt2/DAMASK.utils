#!/bin/bash
rm -f nohup.out; nohup mpirun -np 12 /home/anhtran/Documents/DAMASK/damask-2.0.2//bin/DAMASK_spectral --geom single_phase_equiaxed.geom --load tension.load 2>&1 > log.damask &
