#!/bin/bash
# postResults \
# --cr fp,f,p \
# --split --separation x,y,z \
# --increments \
# --range 10 10 1 singleCrystal_res_50um_tension.spectralOut

cd postProc
# for i in $(seq 10); do
#    addStrainTensors -0 -v singleCrystal_res_50um_tension_inc${i}.txt
#    addCauchy singleCrystal_res_50um_tension_inc${i}.txt
#    addMises -s Cauchy singleCrystal_res_50um_tension_inc${i}.txt
#    addStrainTensors --left --logarithmic singleCrystal_res_50um_tension_inc${i}.txt
#    addMises -e 'ln(V)' singleCrystal_res_50um_tension_inc${i}.txt
#    addDisplacement --nodal singleCrystal_res_50um_tension_inc${i}.txt
# done

for fileName in $(ls singleCrystal_res_50um_tension_inc*.txt); do
    addStrainTensors -0 -v ${fileName}
    addCauchy ${fileName}
    addMises -s Cauchy ${fileName}
    addStrainTensors --left --logarithmic ${fileName}
    addMises -e 'ln(V)' ${fileName}
    addDisplacement --nodal ${fileName}
done

