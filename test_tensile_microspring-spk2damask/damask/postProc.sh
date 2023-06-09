#!/bin/bash

# rm -f nohup.out; nohup bash postProc.sh 2>&1 > log.postProc &

geomFileName="spk_dump_12_out" # ${geomFileName}.geom
loadFileName="tension" # ${loadFileName}.load

# rm -f nohup.out; nohup postResults --cr fp,f,p,grainrotation,texture --split --separation x,y,z --increments --range 1 44 1 ${geomFileName}_${loadFileName}.spectralOut 2>&1 > log.postResults &

cd postProc
for fileName in $(ls -1v ${geomFileName}_${loadFileName}*.txt); do
	addStrainTensors -0 -v ${fileName}
	addCauchy ${fileName}
	addMises -s Cauchy ${fileName}
	addStrainTensors --left --logarithmic ${fileName}
	addMises -e 'ln(V)' ${fileName}
	addDisplacement --nodal ${fileName}
	echo "done processing ${fileName}."
done
