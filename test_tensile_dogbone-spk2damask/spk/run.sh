#!/bin/bash
rm -f *jpg *vti dump*
ln -sf ../in.potts_additive_dogbone .
fakeNProc=$(nproc --all)
nProc=$((${fakeNProc}/2))
mpirun -np 8 spk < in.potts_additive_dogbone
