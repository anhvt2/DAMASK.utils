#!/bin/bash

geomFileName='padded_voidSeeded_1.500pc_spk_dump_12_out'
loadFileName='tension'

cd postProc/
for fileName in $(ls -1v ${geomFileName}_${loadFileName}*.txt); do
    fileName=$(echo ${fileName} | rev | cut -c 5- | rev)
    # fileName=$(echo ${fileName} | cut -d. -f1) # deprecated
    cp ../templatePostProc-1File.sh postProc-${fileName}.sh
    cp ../sbatch.postprocDamask.srn sbatch.postprocDamask-${fileName}.srn
    sed -i "4s|.*|fileName=\"${fileName}\"|" postProc-${fileName}.sh
    sed -i "17s|.*|fileName=\"${fileName}\"|" sbatch.postprocDamask-${fileName}.srn
    # rm -f nohup.out; nohup bash ${fileName}.sh 2>&1 > ${fileName}.sh.log & # execute postProc in background
    echo ${fileName}
done
cd ..

