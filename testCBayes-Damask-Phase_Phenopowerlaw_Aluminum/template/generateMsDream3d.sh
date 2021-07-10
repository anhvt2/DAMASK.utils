#!/bin/bash


## declare parameters
# two params for controlling log-normal distribution
mu=$(cat mu.dat)
sigma=$(cat sigma.dat)
msId=$(cat msId.dat)

dimCell=64


## declare paths

execPath="/ascldap/users/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/bin"
outputPath=$(pwd)

inputFile="test-adoptSingleCubicPhaseEquiaxed-DAMASK.json"
currentPath="${inputPath}"


echo "Running $inputFile"
echo "Output path: $outputPath"
echo
echo "Parameters settings:"
echo "mu = $mu"
echo "sigma = $sigma"
echo "dimCell = $dimCell"
echo




sed -i "53s|.*|                    \"Average\": ${mu},|" ${inputFile}
sed -i "54s|.*|                    \"Standard Deviation\": ${sigma}|" ${inputFile}
sed -i "140s|.*|            \"x\": ${dimCell},|" ${inputFile}
sed -i "141s|.*|            \"y\": ${dimCell},|" ${inputFile}
sed -i "142s|.*|            \"z\": ${dimCell}|" ${inputFile}
sed -i "388s|.*|        \"OutputPath\": \"${outputPath}\"|" ${inputFile}


${execPath}/PipelineRunner -p $(pwd)/${inputFile}

cat material.config.preamble  | cat - material.config | sponge material.config


echo "Microstructure files are generated at:"
echo "$outputPath"
echo
echo

export geom_check=/ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2/processing/pre/geom_check.sh
geom_check *.geom
sh ../getDream3dInfo.sh
