#!/bin/bash

for i in $(seq 0 2807); do
    python3 geom_spk2spk.py --vti="potts_3d.${i}.vti" --phase="phase_dump_12_out.npy"
done
