#!/bin/bash
# set input parameters
qoi="strain" # input parameter
sparse_grid_level=1 # input parameter
dakotaInputFile='textbook5d_uq_sc_pyImport.in' # input file
if [[ "${qoi}" == "strain" ]]; then 
	qoiIndex=0; 
fi
if [[ "${qoi}" == "stress" ]]; then
	qoiIndex=1; 
fi

# parse input
sed -i "7s|.*|    sparse_grid_level = ${sparse_grid_level} # subject to change|" ${dakotaInputFile}
sed -i "42s|.*|o_ = outputData[index_, ${qoiIndex}] # change the second index accordingly: 0 = strainYield, 1 = stressYield|" damask_query.py

# run dakota
../dakota -i "${dakotaInputFile}" > "dakota_${qoi}Yield_level${sparse_grid_level}.log"
grep -inr ' f1' "dakota_${qoi}Yield_level${sparse_grid_level}.log"  > tmp.txt
sed -i  's/ f1//g' tmp.txt
mv tmp.txt ${qoi}Yield_level${sparse_grid_level}.dat


