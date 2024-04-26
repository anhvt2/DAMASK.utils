#!/bin/bash
rm -f *jpg *vti dump*
ln -sf ../in.potts_additive_dogbone .
fakeNProc=$(nproc --all)
nProc=$((${fakeNProc}/2))
mpirun -np ${nProc} spk < in.potts_additive_dogbone
