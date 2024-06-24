#!/bin/bash

# for i in $(seq 1000 100 1500); do
# for i in $(seq $(ls *vti | wc -l)); do
#     python3 geom_spk2spk.py --vti="potts_3d.${i}.vti" --phase="phase_dump_12_out.npy"
# done

for vtiFile in $(ls potts_3d*vti); do
    python3 geom_spk2spk.py --vti="${vtiFile}" --phase="phase_dump_12_out.npy"
done
