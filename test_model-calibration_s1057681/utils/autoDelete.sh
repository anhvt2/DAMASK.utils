#!/bin/bash

# for apBO algorithms
modelName=$(cat modelName.dat)
numOfFolders=$(ls -1dv *_Iter*/ | wc -l)
numInitPoint=3 # get from mainprog.m -- could be smart-read

for i in $(seq $numInitPoint $numOfFolders); do
	rm -rfv ${modelName}_Iter${i}*
done

# for pBO algorithms
rm -rfv ${modelName}_*Ex*Batch*
