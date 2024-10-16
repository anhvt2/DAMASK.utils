#!/bin/bash

for i in $(seq 500); do
    folderName="res-50um-additive-run-${i}"
    cd $folderName
    # cat material.config.preamble  | cat - material.config | sponge material.config
    cp ../material.config.preamble ./material.config
    cat material_void.config >> material.config
    cd ..
    echo "done $folderName"
done
