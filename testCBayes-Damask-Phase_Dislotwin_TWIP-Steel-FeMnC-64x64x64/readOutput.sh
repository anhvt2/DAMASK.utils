#!/bin/bash
for fN in $(ls -1dv */); do
	cd $fN
	if [ -e "postProc/youngModulus.out" ]; then

		o=$(cat postProc/youngModulus.out)
		echo "$fN: output = ${o}"

	fi
	cd ..
done

