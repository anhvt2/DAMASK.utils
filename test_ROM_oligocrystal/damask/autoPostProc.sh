#!/bin/bash
for i in $(seq 1000); do
    cd $i
    cp ../sbatch.postProcess.srn .
    rm -v sbatch.damask.srn
    ssubmit
    echo "done folder $i"
    cd ..
done
