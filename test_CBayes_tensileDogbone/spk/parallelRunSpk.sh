#!/bin/bash

# for i in $(seq -f "%03g" 1 100); do
for i in $(seq 1 100); do
	folderName="res-50um-run-${i}"
	# cp -rfv res-50um/ $folderName
	cd $folderName
	sed -i "3s|.*|seed         ${i}|" in.potts_3d
	cp ../sbatch.spparks.srn .
	rm dump*.out *.vti
	ssubmit
	cd ..
done

