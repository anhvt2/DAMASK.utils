#!/bin/bash

postResults single_phase_equiaxed_tension.spectralOut --cr f,p

cd postProc/
addStrainTensors single_phase_equiaxed_tension.txt --left --logarithmic
addCauchy single_phase_equiaxed_tension.txt
addMises single_phase_equiaxed_tension.txt --strain 'ln(V)' --stress Cauchy
filterTable < single_phase_equiaxed_tension.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > stress_strain.log


