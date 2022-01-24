#!/bin/bash

## require: "sudo apt install moreutils" for "sponge commands"
## run as "bash ./generateMsDream3d.sh"

## declare parameters
# two params for controlling log-normal distribution
mu=$(cat mu.dat)
sigma=$(cat sigma.dat)
msId=$(cat msId.dat)

dimCell=$(cat dimCell.dat)


## declare paths
hostName="$(echo $(hostname))"
if [[ ${hostName} == *"solo"* ]]; then
	execPath="/ascldap/users/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/bin" # solo
elif [[ ${hostName} == *"ideapad"* ]]; then
	execPath="/home/anhvt89/Documents/DREAM.3D/DREAM3D-6.5.128-Linux-x86_64/bin" # ideapad
elif [[ ${hostName} == *"s1057681"* ]]; then
	execPath="/home/anhtran/Documents/DREAM.3D/DREAM3D-6.5.141-Linux-x86_64/bin" # s1057681
elif [[ ${hostName} == *"strix"* ]]; then
	execPath="/home/anhvt89/Documents/DREAM.3D/DREAM3D-6.5.141-Linux-x86_64/bin" # Asus ROG strix
else
	execPath="/home/anhvt89/Documents/DREAM.3D/DREAM3D-6.5.128-Linux-x86_64/bin" # else
fi

outputPath=$(pwd)

# inputFile="test-DownSamplingSVEs-NonExact-base320.json"
# inputFile="PRISMS_pipeline_hcp.json"
inputFile="test-Magnesium.json"
currentPath="${inputPath}"


echo "Running $inputFile"
echo "Output path: $outputPath"
echo
# echo "Parameters settings:"
# echo "mu = $mu"
# echo "sigma = $sigma"
# echo "dimCell = $dimCell"
# echo




# sed -i "47s|.*|                    \"Average\": ${mu},|" ${inputFile}
# sed -i "48s|.*|                    \"Standard Deviation\": ${sigma}|" ${inputFile}
# sed -i "363s|.*|            \"x\": ${dimCell},|" ${inputFile}
# sed -i "364s|.*|            \"y\": ${dimCell},|" ${inputFile}
# sed -i "365s|.*|            \"z\": ${dimCell}|" ${inputFile}


## NOTE: This action could be automated by (1) searching for the line number with this pattern and (2) replacing the found line number in a FOR loop
# sed -i "388s|.*|        \"OutputPath\": \"${outputPath}/72x72x72\"|" ${inputFile}
# sed -i "445s|.*|        \"OutputPath\": \"${outputPath}/60x60x60\"|" ${inputFile}
# sed -i "502s|.*|        \"OutputPath\": \"${outputPath}/48x48x48\"|" ${inputFile}
# sed -i "559s|.*|        \"OutputPath\": \"${outputPath}/36x36x36\"|" ${inputFile}
# sed -i "616s|.*|        \"OutputPath\": \"${outputPath}/24x24x24\"|" ${inputFile}
# sed -i "673s|.*|        \"OutputPath\": \"${outputPath}/18x18x18\"|" ${inputFile}
# sed -i "730s|.*|        \"OutputPath\": \"${outputPath}/12x12x12\"|" ${inputFile}

# sed -i "641s|.*|        \"OutputPath\": \"${outputPath}/16x16x16\"|" ${inputFile}
# sed -i "698s|.*|        \"OutputPath\": \"${outputPath}/8x8x8\"|" ${inputFile}

# replace OutputPath as in the current directory
# defaultPath="/home/anhvt89/Documents/DAMASK/DAMASK.utils/test_MLMC_template/"
# defaultPath=$(grep -inr 'OutputPath' ${inputFile}.json  | head -n 1  | cut -d: -f3 | cut -c 3- | rev | cut -c 10- | rev)
## Purposes: trim quotation marks, spaces, and commas
defaultPath=$(grep -inr 'OutputPath' ${inputFile}.json  | head -n 1  | cut -d: -f3 | cut -c 3- | rev | cut -c 3- | rev)

# convert from 
# 641:        "OutputPath": "/qscratch/anhtran/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs/64x64x64"
# to 
# /qscratch/anhtran/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs/



# sed -i "s|${defaultPath}|${outputPath}/|g" ${inputFile} # add "/" behind ${outputPath}


${execPath}/PipelineRunner -p $(pwd)/${inputFile}


echo "Microstructure files are generated at:"
echo "$outputPath"
echo
echo

echo "${defaultPath}"
