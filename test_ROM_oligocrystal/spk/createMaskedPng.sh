#!/bin/bash

rm masked_potts_3d.*.png

# for i in $(seq 0 516); do
# for i in $(seq 1000 100 1500); do
# for i in $(seq $(ls *vti | wc -l)); do
#     python3 plotms3d_maskedDogbone.py --fileName="masked_potts_3d.${i}.vti"
# done

for maskedvtiFile in $(ls masked_potts_3d*vti); do
    python3 plotms3d_maskedDogbone.py --fileName="${maskedvtiFile}"
done
