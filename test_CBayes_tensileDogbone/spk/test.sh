#!/bin/bash

# for i in $(seq -f "%03g" 1 100); do
# for i in $(seq 338 500); do
for i in $(seq 500); do
	folderName="res-50um-additive-run-${i}"
	cd $folderName

	if [ ! -e "dump.additive_dogbone.2807" ]; then
	    echo ${i}
	fi

#	rm void+phase_dump_12_out.npy

	cd ..
done

