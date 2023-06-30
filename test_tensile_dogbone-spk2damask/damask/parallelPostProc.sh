#!/bin/bash

geomFileName="spk_dump_20_out" # ${geomFileName}.geom
loadFileName="tension" # ${loadFileName}.load

for fileName in $(ls -1v ${geomFileName}_${loadFileName}*.txt); do
    echo ${fileName}
done
    
