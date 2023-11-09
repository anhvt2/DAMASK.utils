#!/bin/bash

for fN in $(ls -1dv *sve-*/); do
	cd $fN

#	if [ ! -e "postProc/youngModulus.out"  ]; then
#		echo "$fN/postProc/youngModulus.out does not exist."
#	fi

	sh ../getDream3dInfo.sh
	cp ../computeYoungModulus.py .

	if [ -e "postProc/youngModulus.out" ]; then
		cd postProc/
		python3 ../computeYoungModulus.py
		cd ..
	fi


	echo "done $fN"
	cd ..
done
