#!/bin/bash

# type in any *.{outputConstitutive,outputCrystallite,outputHomogenization,C_ref,sta,spectralOut}
fileName=$(echo $1 | cut -d"." -f 1)

postResults $fileName.spectralOut --cr f,p

cd postProc/
addStrainTensors $fileName.txt --left --logarithmic
addCauchy $fileName.txt
addMises $fileName.txt --strain 'ln(V)' --stress Cauchy
filterTable < $fileName.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > stress_strain.log


