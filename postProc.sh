#!/bin/bash

postResults $1.spectralOut --cr f,p

cd postProc/
addStrainTensors $1.txt --left --logarithmic
addCauchy $1.txt
addMises $1.txt --strain 'ln(V)' --stress Cauchy
filterTable < $1.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > stress_strain.log


