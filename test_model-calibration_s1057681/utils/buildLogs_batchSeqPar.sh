#!/bin/bash

modelName=$(cat modelName.dat)
numOfIter=$(ls -1dv ${modelName}_Iter*/ | wc -l)

echo 
echo "building global input/output.dat file for analysis"
echo "list; input, output, feasible, complete, rewards, batchID, folder, acquisitionScheme"
echo "numOfIter = $numOfIter"

echo; echo;

# rm -fv postproc.{input,output,feasible,complete,rewards,batchID,folder,acquisitionScheme}.dat
rm -fv postproc.input.dat
rm -fv postproc.output.dat
rm -fv postproc.feasible.dat
rm -fv postproc.complete.dat
rm -fv postproc.rewards.dat
rm -fv postproc.batchID.dat
rm -fv postproc.folder.dat
rm -fv postproc.acquisitionScheme.dat
rm -fv postproc.startTime.dat
rm -fv postproc.stopTime.dat

for folderName in $(ls -1dv *_Iter*/); do
# for i in $(seq $numOfIter); do
	# cat ${folderName}/input.dat | paste -s  >> postproc.input.dat # transpose column vector to row vector
	cat ${folderName}/input.dat >> postproc.input.dat

	for fileType in output feasible complete; do
		if [ -e ${folderName}/${fileType}.dat ]; then
			cat ${folderName}/${fileType}.dat >> postproc.${fileType}.dat
		else
			echo >> postproc.${fileType}.dat
		fi
		# for acquisitionScheme.dat insert '\n'
	done

	# get time stamps
	sed -n 2p ${folderName}/query.log >> postproc.startTime.dat
	sed -n 4p ${folderName}/query.log >> postproc.stopTime.dat
	
	echo "${folderName}" >> postproc.folder.dat
	echo "done ${folderName}/$numOfIter"
done

