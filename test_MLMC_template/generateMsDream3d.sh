#!/bin/bash


## declare parameters
# two params for controlling log-normal distribution
mu=$(cat mu.dat)
sigma=$(cat sigma.dat)
msId=$(cat msId.dat)

dimCell=$(cat dimCell.dat)


## declare paths
hostName=$(hostname)
if [[ ${hostName} == *"solo"* ]]; then
	execPath="/ascldap/users/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/bin"
elif [[ ${hostName} == *"ideapad"* ]]; then
	execPath="/home/anhvt89/Documents/DREAM.3D/DREAM3D-6.5.128-Linux-x86_64/bin" # ideapad or personal computer elsewhere
else
	execPath="/home/anhvt89/Documents/DREAM.3D/DREAM3D-6.5.128-Linux-x86_64/bin" # ideapad or personal computer elsewhere
fi

outputPath=$(pwd)

inputFile="test-DownSamplingSVEs-ideapad320-NonExact-base320.json"
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


## NOTE: This action could be automated by (1) searching for the line number with this pattern and (2) replacing the found line number in a FOR loop
# sed -i "388s|.*|        \"OutputPath\": \"${outputPath}/72x72x72\"|" ${inputFile}
# sed -i "445s|.*|        \"OutputPath\": \"${outputPath}/60x60x60\"|" ${inputFile}
# sed -i "502s|.*|        \"OutputPath\": \"${outputPath}/48x48x48\"|" ${inputFile}
# sed -i "559s|.*|        \"OutputPath\": \"${outputPath}/36x36x36\"|" ${inputFile}
# sed -i "616s|.*|        \"OutputPath\": \"${outputPath}/24x24x24\"|" ${inputFile}
# sed -i "673s|.*|        \"OutputPath\": \"${outputPath}/18x18x18\"|" ${inputFile}
# sed -i "730s|.*|        \"OutputPath\": \"${outputPath}/12x12x12\"|" ${inputFile}

sed -i "651s|.*|        \"OutputPath\": \"${outputPath}/16x16x16\"|" ${inputFile}
sed -i "708s|.*|        \"OutputPath\": \"${outputPath}/8x8x8\"|" ${inputFile}


${execPath}/PipelineRunner -p $(pwd)/${inputFile}


echo "Microstructure files are generated at:"
echo "$outputPath"
echo
echo

export geom_check=/ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2/processing/pre/geom_check.sh
# for dimCell in 72 60 48 36 24 18 12; do
for dimCell in 16 8; do
	cd ${dimCell}x${dimCell}x${dimCell}

	cat ../material.config.preamble  | cat - material.config | sponge material.config
	geom_check *.geom
	sh ../getDream3dInfo.sh
	
	cd ..
done

