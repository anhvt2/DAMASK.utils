#!/bin/bash
# rm -f nohup.out; nohup bash ${fileName}.sh 2>&1 > ${fileName}.sh.log &

geomFileName="spk_dump_20_out" # ${geomFileName}.geom
loadFileName="tension" # ${loadFileName}.load


fileName=$(echo ${fileName} | cut -d. -f1)

addStrainTensors -0 -v ${fileName}.txt
addCauchy ${fileName}.txt
addMises -s Cauchy ${fileName}.txt
addStrainTensors --left --logarithmic ${fileName}.txt
addMises -e 'ln(V)' ${fileName}.txt
addDisplacement --nodal ${fileName}.txt

vtk_rectilinearGrid ${fileName}.txt

vtk_addRectilinearGridData \
 --inplace \
 --data '1_fp','2_fp','3_fp','4_fp','5_fp','6_fp','7_fp','8_fp','9_fp','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p','1_eulerangles','2_eulerangles','3_eulerangles','1_grainrotation','2_grainrotation','3_grainrotation','4_grainrotation','texture','1_ln(V)','2_ln(V)','3_ln(V)','4_ln(V)','5_ln(V)','6_ln(V)','7_ln(V)','8_ln(V)','9_ln(V)','1_Cauchy','2_Cauchy','3_Cauchy','4_Cauchy','5_Cauchy','6_Cauchy','7_Cauchy','8_Cauchy','9_Cauchy','Mises(Cauchy)','1_ln(V)','2_ln(V)','3_ln(V)','4_ln(V)','5_ln(V)','6_ln(V)','7_ln(V)','8_ln(V)','9_ln(V)','Mises(ln(V))' \
 --vtk "${fileName}_pos(cell).vtr" \
 ${fileName}.txt

echo "done processing ${fileName}."

