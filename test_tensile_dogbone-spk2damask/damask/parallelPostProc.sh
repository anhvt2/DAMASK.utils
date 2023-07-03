#!/bin/bash

geomFileName="spk_dump_20_out" # ${geomFileName}.geom
loadFileName="tension" # ${loadFileName}.load

cd postProc/
for fileName in $(ls -1v ${geomFileName}_${loadFileName}*.txt); do
    fileName=$(echo ${fileName} | cut -d. -f1)
    cp ../templatePostProc-1File.sh postProc-${fileName}.sh
    sed -i "4s|.*|fileName=\"${fileName}\"|" postProc-${fileName}.sh
    # rm -f nohup.out; nohup bash ${fileName}.sh 2>&1 > ${fileName}.sh.log & # execute postProc in background
    echo ${fileName}
done
cd ..

