#!/bin/bash


## declare parameters
# two params for controlling log-normal distribution
mu=$(cat mu.dat)
sigma=$(cat sigma.dat)
msId=$(cat msId.dat)

dimCell=64


## declare paths

if [ "$(hostname)" = "strix" ]; then
    execPath="/home/anhvt89/Documents/DREAM.3D/DREAM3D-6.5.141-Linux-x86_64/bin"
elif [ "$(hostname)" = "s1057681" ]; then
    execPath="/home/anhtran/Documents/DREAM.3D/DREAM3D-6.5.141-Linux-x86_64/bin"
else
    execPath="/ascldap/users/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/bin"
fi

outputPath=$(pwd)

inputFile="damask.json"
currentPath="${inputPath}"


echo "Running $inputFile"
echo "Output path: $outputPath"
echo
echo "Parameters settings:"
echo "mu = $mu"
echo "sigma = $sigma"
echo "dimCell = $dimCell"
echo


${execPath}/PipelineRunner -p $(pwd)/${inputFile}

# cat material.config.preamble  | cat - material.config | sponge material.config


# echo "Microstructure files are generated at:"
# echo "$outputPath"
# echo
# echo

# geom_check *.geom
# bash getDream3dInfo.sh
