#!/bin/bash

geomFileName="spk_dump_12_out" # ${geomFileName}.geom
loadFileName="tension" # ${loadFileName}.load

rm -f nohup.out; nohup postResults --cr fp,f,p,grainrotation,texture --split --separation x,y,z --increments --range 1 44 1 ${geomFileName}_${loadFileName}.spectralOut 2>&1 > log.postResults &

cd postProc
for fileName in $(ls -1v ${geomFileName}_${loadFileName}*.txt); do
	addStrainTensors -0 -v ${fileName}.txt
	addCauchy ${fileName}.txt
	addMises -s Cauchy ${fileName}.txt
	addStrainTensors --left --logarithmic ${fileName}.txt
	addMises -e 'ln(V)' ${fileName}.txt
	addDisplacement --nodal ${fileName}.txt
	echo "done processing ${fileName}."
done
