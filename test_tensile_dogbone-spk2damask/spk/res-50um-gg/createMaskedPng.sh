#!/bin/bash

rm masked_potts_3d.*.png

# for i in $(seq 0 516); do
for i in 0 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40 50 60 80 100 150 200 250 300 450 400 550 500; do
    python3 plotms3d_maskedDogbone.py --fileName="masked_potts_3d.${i}.vti"
done
