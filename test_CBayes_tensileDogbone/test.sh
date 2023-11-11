#!/bin/bash
python3 findGaugeLocations.py # dump gaugeFilter.txt
postResults *.spectralOut --cr f,p --filter $(cat gaugeFilter.txt)
cd postProc/
addStrainTensors ${geomFileName}_${loadFileName}.txt --left --logarithmic
addCauchy ${geomFileName}_${loadFileName}.txt
addMises ${geomFileName}_${loadFileName}.txt --strain 'ln(V)' --stress Cauchy
filterTable < ${geomFileName}_${loadFileName}.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > stress_strain.log
cd ..

postResults \
--cr fp,f,p \
--split --separation x,y,z \
--increments \
--range 1 10 1 ${geomFileName}_${loadFileName}.spectralOut

cd postProc
for i in $(seq 10); do
    addStrainTensors -0 -v ${geomFileName}_${loadFileName}_inc${i}.txt
    addCauchy ${geomFileName}_${loadFileName}_inc${i}.txt
    addMises -s Cauchy ${geomFileName}_${loadFileName}_inc${i}.txt
    addStrainTensors --left --logarithmic ${geomFileName}_${loadFileName}_inc${i}.txt
    addMises -e 'ln(V)' ${geomFileName}_${loadFileName}_inc${i}.txt
    addDisplacement --nodal ${geomFileName}_${loadFileName}_inc${i}.txt
done
