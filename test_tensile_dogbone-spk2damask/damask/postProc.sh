#!/bin/bash

rm -f nohup.out; nohup postResults --cr fp,f,p,grainrotation,texture --split --separation x,y,z --increments --range 1 44 1 spk_dump_12_out_tension.spectralOut 2>&1 > log.postResults &

cd postProc
addStrainTensors -0 -v <fileName>.txt
addCauchy <fileName>.txt
addMises -s Cauchy <fileName>.txt
addStrainTensors --left --logarithmic <fileName>.txt
addMises -e 'ln(V)' <fileName>.txt
addDisplacement --nodal <fileName>.txt
