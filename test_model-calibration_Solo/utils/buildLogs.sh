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

for i in $(seq $numOfIter); do
	# cat ${modelName}_Iter${i}/input.dat | paste -s  >> postproc.input.dat # transpose column vector to row vector
	cat ${modelName}_Iter${i}/input.dat >> postproc.input.dat

	for fileType in output feasible complete rewards batchID acquisitionScheme; do
		if [ -e ${modelName}_Iter${i}/${fileType}.dat ]; then
			cat ${modelName}_Iter${i}/${fileType}.dat >> postproc.${fileType}.dat
		else
			echo "nan" >> postproc.${fileType}.dat
		fi
		# for acquisitionScheme.dat insert '\n'
	done
	# special treatment for non-initial sampling
	if [ -f "${modelName}_Iter${i}/acquisitionScheme.dat" ] ; then
		numOfLineDummy=$(cat ${modelName}_Iter${i}/acquisitionScheme.dat | wc -l)
		if [ "${numOfLineDummy}" -eq "0" ]; then
			echo >> postproc.acquisitionScheme.dat
		fi
	else
		# i.e. ${modelName}_Iter${i}/acquisitionScheme.dat does not exist
		echo "-1" >> postproc.acquisitionScheme.dat
	fi

	# get time stamps
	if [ -e ${modelName}_Iter${i}/query.log ]; then # if query.log exists
		numOfLines=$(cat ${modelName}_Iter${i}/query.log | wc -l)
		if [ "$numOfLines" -lt 2 ]; then
			echo "check $(pwd)/${modelName}_Iter${i}/query.log"
			echo >> postproc.startTime.dat
			echo >> postproc.stopTime.dat
		elif [ "$numOfLines" -lt 4 ]; then
			echo "check $(pwd)/${modelName}_Iter${i}/query.log"
			sed -n 2p ${modelName}_Iter${i}/query.log >> postproc.startTime.dat
			echo >> postproc.stopTime.dat
		else
			sed -n 2p ${modelName}_Iter${i}/query.log >> postproc.startTime.dat
			sed -n 4p ${modelName}_Iter${i}/query.log >> postproc.stopTime.dat
		fi
	else # if query.log does not exist, then get from {input,output}.dat
		date -r ${modelName}_Iter${i}/input.dat "+%Y-%m-%d %H:%M:%S" >> postproc.startTime.dat
		date -r ${modelName}_Iter${i}/output.dat "+%Y-%m-%d %H:%M:%S" >> postproc.startTime.dat
	fi

	echo "${modelName}_Iter${i}" >> postproc.folder.dat
	echo "done $i/$numOfIter"
done

sed -i '/^$/d' postproc.acquisitionScheme.dat # remove empty line

