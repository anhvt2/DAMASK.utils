#!/bin/bash
rm -f nohup.out; nohup mpirun -np 16 spk < in.potts_3d 2>&1 > log.spparks &
