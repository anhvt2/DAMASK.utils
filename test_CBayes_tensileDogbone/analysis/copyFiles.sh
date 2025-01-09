#!/bin/bash
mkdir -p train
mkdir -p test
cd train/
for i in $(seq 500); do
    mkdir -p ${i}
    cd ${i}
    cp -fv ../../../spk/res-50um-additive-run-${i}/porosity.txt .
    cp -fv ../../../spk/res-50um-additive-run-${i}/postProc/main_tension.txt .
    cp -fv ../../../spk/res-50um-additive-run-${i}/postProc/stress_strain.log .
    cd ..
done
cd ..

cd test/

for testIdx in $(cat ../testFolders.txt); do
    testFolder="test-run-${testIdx}"
    mkdir -p ${testFolder}
    cd ${testFolder}
    cp -fv ../../../spk/${testFolder}/porosity.txt .
    cp -fv ../../../spk/${testFolder}/postProc/main_tension.txt .
    cp -fv ../../../spk/${testFolder}/postProc/stress_strain.log .
    cd ..
done

cd ..

