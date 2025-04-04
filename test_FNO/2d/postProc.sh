#!/bin/bash

postResults *.spectralOut --cr f,p
cd postProc/
addStrainTensors simple2d_tension.txt --left --logarithmic
addCauchy simple2d_tension.txt
addMises simple2d_tension.txt --strain 'ln(V)' --stress Cauchy
filterTable < simple2d_tension.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > stress_strain.log
cd ..

postResults \
    --cr fp,f,p,grainrotation,texture \
    --split \
    --separation x,y,z \
    --increments \
    --range 1 30 1 simple2d_tension.spectralOut 2>&1 > log.postResults

