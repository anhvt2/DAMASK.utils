#!/bin/bash
../dakota -i textbook16d_uq_sc_pyImport.in > dakota.log
grep -inr ' f1' dakota.log  > tmp.txt
sed -i  's/ f1//g' tmp.txt
mv tmp.txt strainYield_level1.dat
