#!/bin/bash
# https://stackoverflow.com/questions/3811345/how-to-pass-all-arguments-passed-to-my-bash-script-to-a-function-of-mine
# how to use: 
# bash plot.sh ${folderName}
# bash plot.sh 64x64x64 -- ${folderName} without '/'

args=("$@")

# echo Number of arguments: $#
# echo 1st argument: ${args[0]}
# echo 2nd argument: ${args[1]}
# folderName=$(echo "${args[0]}" | rev | cut -c 1- | rev )
folderName=$(echo "${args[0]}")
echo "Plotting in ${folderName}/postProc/"
python3 plotStressStrain.py --StressStrainFile="$(pwd)/${folderName}/postProc/single_phase_equiaxed_${folderName}_tension.txt" --LoadFile="tension.load"

