#!/bin/bash

# for i in $(seq -f "%03g" 1 100); do
# for i in $(seq 501 501); do
for i in $(cat tmp2.txt); do
	folderName="res-50um-additive-run-${i}"
	templateFolderName="res-50um-additive"
#	cp -rfv ${templateFolderName}/ $folderName
	cd $folderName
	rm -rfv res-50um-additive/
	sed -i "3s|.*|seed         ${i}|" in.potts_additive_dogbone # in.potts_3d
	cp ../sbatch.spparks.srn .
	rm -fv dump*.out *.vti
	ssubmit
	cd ..
done

