#!/bin/bash

for j in $(seq 1 10); do
  bash generateMsDream3d.sh
  mv 2x2x2 sve${j}_2x2x2
  mv 4x4x4 sve${j}_4x4x4
  mv 8x8x8 sve${j}_8x8x8
  mv 10x10x10 sve${j}_10x10x10
  mv 16x16x16 sve${j}_16x16x16
  mv 20x20x20 sve${j}_20x20x20
  mv 40x40x40 sve${j}_40x40x40
  mv 80x80x80 sve${j}_80x80x80
done
